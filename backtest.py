"""
Бэктест SMT / Quasimodo системы (та же логика и ядро, что и в main.py).

Walk-forward: цикл идёт по закрытым LTF-барам; main TF «дорисовывается» только
по мере закрытия его баров, свинги подтверждаются с лагом — заглядывания в будущее нет.
Вход — по open следующего LTF-бара после подтверждения; если SL и TP попадают
в один бар, консервативно считается SL.

Учитываются:
  * торговая сессия 02:00–10:00 по Нью-Йорку (config / --no-session /
    --session-start / --session-end; время баров приводится к UTC через
    --bars-utc-offset — смещение сервера брокера, обычно 2 или 3);
  * дневной лимит убытка 3% от баланса на начало дня (--max-daily-loss, 0 = выкл);
  * переключатели стратегий: --no-smt, --no-qm, --sweep-only, --with-trend-only.

Источники данных:
  * mt5 — история из терминала (Windows):
      python backtest.py --symbol XAUUSD --source mt5 --start 2026-02-01 --end 2026-08-01 \
          --model all --bars-utc-offset 3 --plot
  * csv — файлы (любая ОС):
      python backtest.py --symbol XAUUSD --source csv --csv-main data/xau_m15.csv \
          --csv-ltf data/xau_m5.csv --csv-corr data/xag_m15.csv --tf-main M15 --tf-ltf M5 --plot
Результаты: <out>/trades_*.csv, stats_*.json, equity_*.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from config import Settings, TF_MODELS, TF_MINUTES, SMT_MAP
from core.engine import SignalEngine, Signal
from core.data import load_csv
from core.patterns import invert_ohlc
from core.session import bars_to_utc, local_day

log = logging.getLogger("backtest")

START_BALANCE = 10_000.0


# ----------------------------- сделка -----------------------------

@dataclass
class Trade:
    symbol: str
    model: str
    direction: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    sl: float
    tp: float
    exit_price: float
    exit_reason: str          # 'tp' | 'sl' | 'eod'
    rr_planned: float
    r_multiple: float
    pnl_money: float
    balance_after: float
    smt: Optional[bool]
    quasimodo: bool
    trend: str
    comment: str


# ----------------------------- прогон одной модели -----------------------------

class Backtester:
    def __init__(self, symbol: str, model_id: str, tf_main: str, tf_ltf: str,
                 main_df: pd.DataFrame, ltf_df: pd.DataFrame,
                 corr_df: Optional[pd.DataFrame], settings: Settings):
        self.symbol = symbol
        self.model_id = model_id
        self.tf_main = tf_main
        self.tf_ltf = tf_ltf
        self.tf_ltf_min = TF_MINUTES[tf_ltf]
        self.tf_main_min = TF_MINUTES[tf_main]
        self.main = main_df
        self.ltf = ltf_df
        self.corr = corr_df
        self.s = settings
        self.engine = SignalEngine(settings)  # сессия — по времени баров
        self.trades: List[Trade] = []
        # дневной лимит убытка
        self.balance = START_BALANCE
        self.day_key = None
        self.day_start_balance: Optional[float] = None
        self.locked = False
        self.lock_days = 0

    # ---------------- основной цикл ----------------
    def run(self, start_time: Optional[pd.Timestamp] = None) -> List[Trade]:
        s = self.s
        ltf, main = self.ltf, self.main
        main_times = main.index
        mt_pos = 0                      # сколько main-баров уже «закрыто» и передано движку
        open_tr: Optional[dict] = None  # открытая позиция
        pending_sig: Optional[Signal] = None

        start_i = 60
        if start_time is not None:
            start_i = max(start_i, int(ltf.index.searchsorted(start_time)))

        # прогреваем движок main-барами, закрытыми до старта
        warm_now = ltf.index[start_i] if start_i < len(ltf) else ltf.index[-1]
        while mt_pos < len(main_times) and \
                main_times[mt_pos] + pd.Timedelta(minutes=self.tf_main_min) <= warm_now:
            mt_pos += 1
        if mt_pos >= 60:
            self._feed_main(mt_pos)     # сигнал прогрева отбрасывается

        for i in range(start_i, len(ltf)):
            t_open = ltf.index[i]
            bar = ltf.iloc[i]
            now_close = t_open + pd.Timedelta(minutes=self.tf_ltf_min)
            self._roll_day(now_close)

            # 1) исполнение отложенного входа по open текущего бара
            if pending_sig is not None and open_tr is None:
                if self.locked:
                    pending_sig = None
                else:
                    open_tr = self._open_trade(pending_sig, t_open, float(bar["open"]))
                    pending_sig = None

            # 2) сопровождение открытой позиции (SL/TP внутри бара)
            if open_tr is not None:
                open_tr = self._manage(open_tr, t_open, bar)

            # 3) закрытие main-баров по мере наступления их времени закрытия
            new_main = False
            while mt_pos < len(main_times) and \
                    main_times[mt_pos] + pd.Timedelta(minutes=self.tf_main_min) <= now_close:
                mt_pos += 1
                new_main = True
            if new_main and mt_pos >= 60:
                sig_m = self._feed_main(mt_pos)   # режим «только свип»
                if sig_m is not None and open_tr is None \
                        and pending_sig is None and not self.locked:
                    pending_sig = sig_m

            # 4) поиск подтверждения iFVG на закрытии LTF-бара
            key = (self.symbol, self.model_id)
            if key in self.engine.pending and open_tr is None and pending_sig is None:
                window = ltf.iloc[max(0, i + 1 - s.hist_ltf_bars): i + 1]
                sig = self.engine.on_ltf_bar(self.symbol, self.model_id, window)
                if sig is not None and not self.locked:
                    pending_sig = sig

        if open_tr is not None:  # незакрытая позиция в конце теста
            last = self.ltf.iloc[-1]
            self._close(open_tr, self.ltf.index[-1], float(last["close"]), "eod")
        return self.trades

    # ---------------- внутренности ----------------

    def _roll_day(self, bar_close_ts: pd.Timestamp):
        if self.s.max_daily_loss_pct <= 0:
            return
        dk = local_day(bars_to_utc(bar_close_ts, self.s), self.s)
        if dk != self.day_key:
            self.day_key = dk
            self.day_start_balance = self.balance
            self.locked = False

    def _feed_main(self, mt_pos: int) -> Optional[Signal]:
        window = self.main.iloc[max(0, mt_pos - self.s.hist_main_bars): mt_pos]
        t_last = window.index[-1]
        corr_w = None
        if self.corr is not None:
            corr_w = self.corr[self.corr.index <= t_last]
            corr_w = corr_w.iloc[-self.s.hist_main_bars:]
        return self.engine.on_main_bar(self.symbol, self.model_id, self.tf_main,
                                       self.tf_ltf, window, corr_w)

    def _open_trade(self, sig: Signal, t: pd.Timestamp, open_price: float) -> Optional[dict]:
        # пересчитываем риск от фактической цены входа (open следующего бара)
        entry = open_price
        risk = entry - sig.sl if sig.direction == "buy" else sig.sl - entry
        if risk <= 0:
            return None
        reward = sig.tp - entry if sig.direction == "buy" else entry - sig.tp
        if reward <= 0:
            return None
        risk_money = self.balance * self.s.risk_per_trade
        return {"sig": sig, "entry": entry, "time": t, "risk": risk,
                "rr": reward / risk, "risk_money": risk_money}

    def _manage(self, tr: dict, t: pd.Timestamp, bar: pd.Series) -> Optional[dict]:
        sig: Signal = tr["sig"]
        lo, hi = float(bar["low"]), float(bar["high"])
        if sig.direction == "buy":
            hit_sl = lo <= sig.sl
            hit_tp = hi >= sig.tp
        else:
            hit_sl = hi >= sig.sl
            hit_tp = lo <= sig.tp
        if hit_sl and hit_tp and self.s.conservative_fills:
            hit_tp = False
        if hit_sl:
            self._close(tr, t, sig.sl, "sl")
            return None
        if hit_tp:
            self._close(tr, t, sig.tp, "tp")
            return None
        return tr

    def _close(self, tr: dict, t: pd.Timestamp, price: float, reason: str):
        sig: Signal = tr["sig"]
        d = 1.0 if sig.direction == "buy" else -1.0
        pnl_price = (price - tr["entry"]) * d - self.s.spread  # спред за круг
        r = pnl_price / tr["risk"]
        pnl_money = tr["risk_money"] * r
        self.balance += pnl_money
        self.trades.append(Trade(
            symbol=self.symbol, model=self.model_id, direction=sig.direction,
            signal_time=str(sig.time), entry_time=str(tr["time"]), exit_time=str(t),
            entry=round(tr["entry"], 6), sl=round(sig.sl, 6), tp=round(sig.tp, 6),
            exit_price=round(price, 6), exit_reason=reason,
            rr_planned=round(tr["rr"], 2), r_multiple=round(r, 3),
            pnl_money=round(pnl_money, 2), balance_after=round(self.balance, 2),
            smt=sig.smt, quasimodo=sig.quasimodo, trend=sig.trend, comment=sig.comment,
        ))
        # проверка дневного лимита убытка после фиксации результата
        if self.s.max_daily_loss_pct > 0 and self.day_start_balance and not self.locked:
            if self.balance <= self.day_start_balance * (1.0 - self.s.max_daily_loss_pct):
                self.locked = True
                self.lock_days += 1
                log.info("[%s %s] дневной лимит убытка %.1f%% достигнут (%s) — "
                         "входы заблокированы до следующего дня",
                         self.symbol, self.model_id,
                         self.s.max_daily_loss_pct * 100, self.day_key)


# ----------------------------- статистика -----------------------------

def summarize(trades: List[Trade], risk_per_trade: float) -> dict:
    if not trades:
        return {"trades": 0}
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    eq_r = np.cumsum(rs)
    dd = float(np.max(np.maximum.accumulate(eq_r) - eq_r)) if len(eq_r) else 0.0
    gp, gl = float(wins.sum()), float(-losses.sum())
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    be_wr = None  # винрейт безубыточности при текущих средних win/loss
    if avg_win and avg_loss:
        be_wr = round(100.0 * (-avg_loss) / (avg_win - avg_loss), 1)

    def bucket(keyf):
        out = {}
        for t in trades:
            k = str(keyf(t))
            a = out.setdefault(k, {"trades": 0, "total_r": 0.0, "wins": 0})
            a["trades"] += 1
            a["total_r"] = round(a["total_r"] + t.r_multiple, 3)
            a["wins"] += int(t.r_multiple > 0)
        for a in out.values():
            a["winrate_%"] = round(100.0 * a["wins"] / a["trades"], 1)
            a["avg_r"] = round(a["total_r"] / a["trades"], 3)
        return out

    def alignment(t):
        if t.trend not in ("up", "down"):
            return "range"
        with_t = (t.direction == "buy" and t.trend == "up") or \
                 (t.direction == "sell" and t.trend == "down")
        return "with_trend" if with_t else "counter_trend"

    def sorted_bucket(keyf):
        b = bucket(keyf)
        return {k: b[k] for k in sorted(b)}

    return {
        "trades": len(trades),
        "winrate_%": round(100.0 * len(wins) / len(rs), 1),
        "total_R": round(float(rs.sum()), 2),
        "avg_R": round(float(rs.mean()), 3),
        "avg_win_R": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss_R": round(avg_loss, 2) if avg_loss is not None else None,
        "breakeven_winrate_%": be_wr,
        "best_R": round(float(rs.max()), 2),
        "worst_R": round(float(rs.min()), 2),
        "profit_factor": round(gp / gl, 2) if gl > 0 else None,
        "max_drawdown_R": round(dd, 2),
        "risk_per_trade": risk_per_trade,
        "by_year": sorted_bucket(lambda t: t.entry_time[:4]),
        "by_month": sorted_bucket(lambda t: t.entry_time[:7]),
        "by_model": bucket(lambda t: t.model),
        "by_direction": bucket(lambda t: t.direction),
        "by_trend": bucket(lambda t: t.trend),
        "by_alignment": bucket(alignment),
        "by_model_alignment": bucket(lambda t: f"{t.model}|{alignment(t)}"),
        "by_model_qm": bucket(lambda t: f"{t.model}|qm={t.quasimodo}"),
        "by_exit": bucket(lambda t: t.exit_reason),
        "by_smt": bucket(lambda t: t.smt),
        "by_quasimodo": bucket(lambda t: t.quasimodo),
        "by_smt_qm": bucket(lambda t: f"smt={t.smt}|qm={t.quasimodo}"),
    }


def monte_carlo(trades: List[Trade], n_iter: int, risk: float) -> dict:
    """Бутстрэп последовательности сделок: распределение итога и максимальной
    просадки (в R и в % счёта при фикс. доле риска). Отвечает на вопрос
    «какие просадки реалистичны при таком профиле сделок»."""
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    n = len(rs)
    rng = np.random.default_rng(42)
    n_iter = int(min(max(n_iter, 100), 100_000))
    paths = rng.choice(rs, size=(n_iter, n), replace=True)
    eq = np.cumsum(paths, axis=1)
    dds = np.max(np.maximum.accumulate(eq, axis=1) - eq, axis=1)
    finals = eq[:, -1]

    def q(a, p):
        return round(float(np.percentile(a, p)), 2)

    def dd_pct(dd_r):  # просадка в % счёта при фиксированной доле риска
        return round(100.0 * (1.0 - (1.0 - risk) ** dd_r), 1)

    return {
        "iterations": n_iter,
        "trades_per_path": n,
        "total_R": {"p5": q(finals, 5), "p50": q(finals, 50), "p95": q(finals, 95)},
        "prob_loss_%": round(100.0 * float((finals <= 0).mean()), 1),
        "max_drawdown_R": {"p50": q(dds, 50), "p95": q(dds, 95), "p99": q(dds, 99)},
        "max_drawdown_pct_at_current_risk": {
            "p50": dd_pct(q(dds, 50)), "p95": dd_pct(q(dds, 95)),
            "p99": dd_pct(q(dds, 99)),
        },
    }


def save_outputs(trades: List[Trade], stats: dict, out_dir: str, tag: str, plot: bool):
    os.makedirs(out_dir, exist_ok=True)
    tpath = os.path.join(out_dir, f"trades_{tag}.csv")
    spath = os.path.join(out_dir, f"stats_{tag}.json")
    pd.DataFrame([asdict(t) for t in trades]).to_csv(tpath, index=False)
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("Сохранено: %s, %s", tpath, spath)
    if plot and trades:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            eq = np.cumsum([t.r_multiple for t in trades])
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(range(1, len(eq) + 1), eq, lw=1.6)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title(f"Equity (R) — {tag}")
            ax.set_xlabel("Сделка")
            ax.set_ylabel("Накопленный R")
            ax.grid(alpha=0.3)
            ppath = os.path.join(out_dir, f"equity_{tag}.png")
            fig.tight_layout()
            fig.savefig(ppath, dpi=130)
            plt.close(fig)
            log.info("График: %s", ppath)
        except Exception as e:  # matplotlib может отсутствовать
            log.warning("График не построен: %s", e)


# ----------------------------- запуск -----------------------------

def apply_args(s: Settings, args) -> Settings:
    if args.no_smt:
        s.use_smt = False
    if args.no_qm:
        s.use_quasimodo = False
    if args.require_qm:
        s.require_qm = True
    if args.no_qm_substitute:
        s.qm_substitutes_smt = False
    if args.sweep_only:
        s.require_ifvg = False
    if args.smt_mode:
        s.smt_mode = args.smt_mode
    if args.with_trend_only:
        s.with_trend_only = True
    if args.no_session:
        s.session_enabled = False
    if args.session_start:
        s.session_start = args.session_start
    if args.session_end:
        s.session_end = args.session_end
    if args.bars_utc_offset is not None:
        s.bars_utc_offset_hours = args.bars_utc_offset
    if args.max_daily_loss is not None:
        s.max_daily_loss_pct = args.max_daily_loss
    if args.min_rr is not None:
        s.min_rr = args.min_rr
    if args.spread is not None:
        s.spread = args.spread
    if args.risk is not None:
        s.risk_per_trade = args.risk
    return s


def run_one(symbol, tf_main, tf_ltf, main_df, ltf_df, corr_df, s, start,
            out_dir, plot, mc: int = 0) -> List[Trade]:
    mid = f"{tf_main}/{tf_ltf}"
    bt = Backtester(symbol, mid, tf_main, tf_ltf, main_df, ltf_df, corr_df, s)
    trades = bt.run(start)
    st = summarize(trades, s.risk_per_trade)
    st["end_balance_from_10k"] = round(bt.balance, 2)
    st["daily_loss_lock_days"] = bt.lock_days
    if mc and trades:
        st["monte_carlo"] = monte_carlo(trades, mc, s.risk_per_trade)
    log.info("Модель %s: сделок=%s, total_R=%s, баланс=%s, дней с лимитом=%s",
             mid, st.get("trades"), st.get("total_R"),
             st["end_balance_from_10k"], bt.lock_days)
    save_outputs(trades, st, out_dir, f"{symbol}_{mid.replace('/', '-')}", plot)
    return trades


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser(description="Бэктест SMT/Quasimodo системы")
    p.add_argument("--symbol", required=True)
    p.add_argument("--source", choices=["mt5", "csv"], default="csv")
    p.add_argument("--start", help="YYYY-MM-DD (для source=mt5 и/или отсечки старта)")
    p.add_argument("--end", help="YYYY-MM-DD (source=mt5)")
    p.add_argument("--model", default="all",
                   help="'all' | индекс модели 0..%d | вид M15/M5" % (len(TF_MODELS) - 1))
    p.add_argument("--csv-main", help="CSV main TF (source=csv)")
    p.add_argument("--csv-ltf", help="CSV LTF (source=csv)")
    p.add_argument("--csv-corr", help="CSV коррелята на main TF (для SMT), опционально")
    p.add_argument("--tf-main", default="M15")
    p.add_argument("--tf-ltf", default="M5")
    # переключатели стратегий
    p.add_argument("--no-smt", action="store_true")
    p.add_argument("--no-qm", action="store_true")
    p.add_argument("--require-qm", action="store_true",
                   help="входы только при Quasimodo-конфлюэнсе")
    p.add_argument("--no-qm-substitute", action="store_true",
                   help="QM не заменяет SMT (для строгого конфлюэнса SMT+QM)")
    p.add_argument("--sweep-only", action="store_true",
                   help="входы сразу по свипу, без iFVG на LTF")
    p.add_argument("--smt-mode", choices=["prefer", "require", "require_if_available"])
    p.add_argument("--with-trend-only", action="store_true")
    # сессия / лимиты
    p.add_argument("--no-session", action="store_true")
    p.add_argument("--session-start")
    p.add_argument("--session-end")
    p.add_argument("--bars-utc-offset", type=float,
                   help="смещение времени баров от UTC (сервер брокера, напр. 3)")
    p.add_argument("--max-daily-loss", type=float, help="0.03 = 3%%; 0 = выкл")
    # риск / издержки
    p.add_argument("--min-rr", type=float)
    p.add_argument("--spread", type=float, help="спред в ценовых единицах (напр. 0.20 для XAUUSD)")
    p.add_argument("--risk", type=float, help="доля риска на сделку (для денег/лимита)")
    p.add_argument("--out", default="bt_results")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--mc", type=int, default=0,
                   help="Monte-Carlo: число итераций бутстрэпа (напр. 10000; 0 = выкл)")
    args = p.parse_args()

    s = apply_args(Settings(), args)
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None

    log.info("Настройки: SMT=%s QM=%s iFVG=%s | сессия %s %s-%s (%s, offset=%s) | "
             "дневной лимит=%s | risk=%.2f%%",
             s.use_smt and s.smt_mode, s.use_quasimodo, s.require_ifvg,
             "вкл" if s.session_enabled else "ВЫКЛ", s.session_start, s.session_end,
             s.session_tz, s.bars_utc_offset_hours,
             f"{s.max_daily_loss_pct:.1%}" if s.max_daily_loss_pct > 0 else "выкл",
             s.risk_per_trade * 100)

    # инверсия коррелята по конфигу (EURUSD -> iDXY)
    invert_corr = False
    for base, cfg in SMT_MAP.items():
        if args.symbol.upper().startswith(base):
            invert_corr = cfg["invert"]
            break

    all_trades: List[Trade] = []
    if args.source == "csv":
        if not (args.csv_main and args.csv_ltf):
            p.error("для source=csv нужны --csv-main и --csv-ltf")
        main_df = load_csv(args.csv_main)
        ltf_df = load_csv(args.csv_ltf)
        corr_df = load_csv(args.csv_corr) if (args.csv_corr and s.use_smt) else None
        if corr_df is not None and invert_corr:
            corr_df = invert_ohlc(corr_df)
            log.info("Коррелят инвертирован (iDXY)")
        all_trades += run_one(args.symbol, args.tf_main, args.tf_ltf,
                              main_df, ltf_df, corr_df, s, start, args.out,
                              args.plot, args.mc)
        models_n = 1
    else:
        if start is None or end is None:
            p.error("для source=mt5 нужны --start и --end")
        from core.mt5_client import MT5Client
        client = MT5Client()
        if not client.connect():
            sys.exit(1)

        if args.model == "all":
            models = [m for m in TF_MODELS if m.get("enabled", True)]
        elif "/" in args.model:
            mmain, mltf = args.model.split("/")
            models = [{"main": mmain, "ltf": mltf}]
        else:
            models = [TF_MODELS[int(args.model)]]
        models_n = len(models)

        corr_name = None
        if s.use_smt:
            for base, cfg in SMT_MAP.items():
                if args.symbol.upper().startswith(base):
                    from config import DXY_ALIASES, SILVER_ALIASES, GOLD_ALIASES
                    alias_map = {"DXY": DXY_ALIASES, "XAGUSD": SILVER_ALIASES,
                                 "XAUUSD": GOLD_ALIASES}
                    corr_name = client.resolve_symbol(
                        alias_map.get(cfg["corr"], [cfg["corr"]]))
                    break

        def fetch(sym, tf):
            df = client.rates_range(sym, tf, start - pd.Timedelta(days=12), end)
            if df is not None:
                df.index = df.index.tz_localize(None)
            return df

        for model in models:
            main_df = fetch(args.symbol, model["main"])
            ltf_df = fetch(args.symbol, model["ltf"])
            if main_df is None or ltf_df is None:
                log.error("Нет истории %s для модели %s/%s",
                          args.symbol, model["main"], model["ltf"])
                continue
            corr_df = fetch(corr_name, model["main"]) if corr_name else None
            if corr_df is not None and invert_corr:
                corr_df = invert_ohlc(corr_df)
            all_trades += run_one(args.symbol, model["main"], model["ltf"],
                                  main_df, ltf_df, corr_df, s, start,
                                  args.out, args.plot, args.mc)
        client.shutdown()

    if models_n > 1 and all_trades:
        all_trades.sort(key=lambda t: t.entry_time)
        st = summarize(all_trades, s.risk_per_trade)
        if args.mc:
            st["monte_carlo"] = monte_carlo(all_trades, args.mc, s.risk_per_trade)
        save_outputs(all_trades, st, args.out, f"{args.symbol}_ALL", args.plot)
        log.info("ИТОГО по всем моделям: %s", json.dumps(st, ensure_ascii=False, indent=2))
    elif all_trades:
        log.info("Итог: %s", json.dumps(summarize(all_trades, s.risk_per_trade),
                                        ensure_ascii=False, indent=2))
    else:
        log.info("Сделок не найдено — проверьте период, сессию, режим SMT и настройки.")


if __name__ == "__main__":
    main()
