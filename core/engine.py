"""Ядро сигналов (общее для realtime и бэктеста).

Поток:
  on_main_bar()  — структура, тренд, пулы внешней ликвидности, двухсвечной свип,
                   SMT-дивергенция с коррелятом, конфлюэнс Quasimodo.
                   При require_ifvg=True создаётся pending-сетап (ждём LTF),
                   при require_ifvg=False сигнал отдаётся сразу («только свип»).
  on_ltf_bar()   — ожидание инверсии FVG (полное закрепление телом против имбаланса)
                   в окне подтверждения -> Signal (entry/SL/TP).

Переключатели (Settings): use_smt, use_quasimodo, require_ifvg, with_trend_only.
Торговая сессия: сигналы отдаются только внутри окна session_start..session_end
по session_tz (в realtime — по системным часам через now_fn, в бэктесте — по
времени бара с учётом bars_utc_offset_hours).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import pandas as pd

from config import Settings, TF_MINUTES
from .structure import analyze_structure, atr_value
from .patterns import find_recent_sweep, find_first_inversion, smt_check
from .session import bars_to_utc, in_session

log = logging.getLogger("engine")


@dataclass
class Pending:
    symbol: str
    model_id: str
    tf_ltf: str
    direction: str            # 'buy' | 'sell'
    side: str                 # 'low' | 'high'
    level: float
    extreme: float
    pierce_time: pd.Timestamp
    sweep_time: pd.Timestamp  # время реакционной свечи
    deadline: pd.Timestamp
    smt: Optional[bool]
    quasimodo: bool
    trend: str
    pool_low: float
    pool_high: float
    atr_main: float
    scanned_until: Optional[pd.Timestamp] = None  # до какого LTF-времени уже искали iFVG


@dataclass
class Signal:
    symbol: str
    model_id: str
    direction: str
    time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    rr: float
    smt: Optional[bool]
    quasimodo: bool
    trend: str
    comment: str


class SignalEngine:
    def __init__(self, settings: Settings,
                 now_fn: Optional[Callable[[], pd.Timestamp]] = None):
        self.s = settings
        self.now_fn = now_fn  # realtime: lambda: pd.Timestamp.now(tz="UTC")
        self.pending: Dict[Tuple[str, str], Pending] = {}
        self._seen: Set[Tuple[str, str, str, str]] = set()

    # ------------------------- сессия -------------------------
    def _session_ok(self, bar_close_time) -> bool:
        s = self.s
        if not s.session_enabled:
            return True
        ts = self.now_fn() if self.now_fn is not None else bars_to_utc(bar_close_time, s)
        return in_session(ts, s)

    # ------------------------- MAIN TF -------------------------
    def on_main_bar(self, symbol: str, model_id: str, tf_main: str, tf_ltf: str,
                    main_df: pd.DataFrame,
                    corr_df: Optional[pd.DataFrame]) -> Optional[Signal]:
        s = self.s
        if main_df is None or len(main_df) < 60:
            return None
        key = (symbol, model_id)
        st = analyze_structure(main_df, s)
        if st.pools.low is None or st.pools.high is None:
            return None

        # инвалидация pending: закреп телом за уровнем или истёк дедлайн
        p = self.pending.get(key)
        if p is not None:
            last_close = float(main_df["close"].iloc[-1])
            now = main_df.index[-1]
            broke = (p.side == "low" and last_close < p.level) or \
                    (p.side == "high" and last_close > p.level)
            if broke or now > p.deadline:
                log.info("[%s %s] pending отменён (%s)", symbol, model_id,
                         "закреп за уровнем" if broke else "истёк дедлайн")
                self.pending.pop(key, None)

        sweep = find_recent_sweep(main_df, st.pools, s.max_sweep_age,
                                  st.trend, s.with_trend_only)
        if sweep is None:
            return None
        uid = (symbol, model_id, sweep.side, str(sweep.pierce_time))
        if uid in self._seen:
            return None
        self._seen.add(uid)

        direction = "buy" if sweep.side == "low" else "sell"
        level_swing = st.pools.low if sweep.side == "low" else st.pools.high
        tf_min = TF_MINUTES[tf_main]

        # SMT: коррелят снял / не снял соответствующую ликвидность
        eff_smt = s.smt_mode if s.use_smt else "off"
        smt = None
        if eff_smt != "off":
            max_ref = pd.Timedelta(minutes=tf_min * s.smt_ref_window)
            smt = smt_check(corr_df, sweep, level_swing.time,
                            s.swing_left, s.swing_right, max_ref)

        # Quasimodo-конфлюэнс: хвост свипа в зоне QML..head подходящего QM
        qm = False
        if s.use_quasimodo or s.require_qm:
            qm = any(q.direction == direction and
                     min(q.qml, q.head) <= sweep.extreme <= max(q.qml, q.head)
                     for q in st.qm_levels)
        qm_sub = s.use_quasimodo and s.qm_substitutes_smt and qm

        if s.require_qm and not qm:
            log.info("[%s %s] свип %s @ %.5f отклонён: нет Quasimodo-конфлюэнса "
                     "(require_qm)", symbol, model_id, sweep.side, sweep.level)
            return None

        ok = True
        if eff_smt == "require":
            ok = (smt is True) or qm_sub
        elif eff_smt == "require_if_available":
            ok = (smt is not False) or qm_sub
        if not ok:
            log.info("[%s %s] свип %s @ %.5f отклонён: нет SMT/QM (smt=%s qm=%s)",
                     symbol, model_id, sweep.side, sweep.level, smt, qm)
            return None

        # ---- режим «только свип»: вход без LTF-подтверждения ----
        if not s.require_ifvg:
            bar_close = main_df.index[-1] + pd.Timedelta(minutes=tf_min)
            if not self._session_ok(bar_close):
                log.info("[%s %s] свип %s @ %.5f вне торговой сессии — вход пропущен",
                         symbol, model_id, sweep.side, sweep.level)
                return None
            entry = float(main_df["close"].iloc[-1])
            sig = self._finalize(symbol, model_id, direction, sweep.extreme,
                                 sweep.extreme, entry, float(st.pools.low.price),
                                 float(st.pools.high.price), st.atr, smt, qm,
                                 st.trend, main_df.index[-1])
            if sig:
                log.info("[%s %s] СИГНАЛ ПО СВИПУ (без iFVG) %s: entry=%.5f sl=%.5f "
                         "tp=%.5f rr=%.2f", symbol, model_id,
                         sig.direction.upper(), sig.entry, sig.sl, sig.tp, sig.rr)
            return sig

        # ---- стандартный режим: ждём iFVG на LTF ----
        deadline = sweep.time + pd.Timedelta(minutes=tf_min * (s.confirm_window_main + 1))
        self.pending[key] = Pending(
            symbol=symbol, model_id=model_id, tf_ltf=tf_ltf,
            direction=direction, side=sweep.side,
            level=sweep.level, extreme=sweep.extreme,
            pierce_time=sweep.pierce_time, sweep_time=sweep.time, deadline=deadline,
            smt=smt, quasimodo=qm, trend=st.trend,
            pool_low=float(st.pools.low.price), pool_high=float(st.pools.high.price),
            atr_main=st.atr,
        )
        log.info("[%s %s] СВИП %s @ %.5f (тренд=%s, SMT=%s, QM=%s) -> ждём iFVG до %s",
                 symbol, model_id, sweep.side, sweep.level, st.trend, smt, qm, deadline)
        return None

    # ------------------------- LTF -------------------------
    def on_ltf_bar(self, symbol: str, model_id: str,
                   ltf_df: pd.DataFrame) -> Optional[Signal]:
        s = self.s
        key = (symbol, model_id)
        p = self.pending.get(key)
        if p is None or ltf_df is None or len(ltf_df) < 10:
            return None
        now = ltf_df.index[-1]
        if now > p.deadline:
            self.pending.pop(key, None)
            return None

        # вне торговой сессии входов нет; инверсии этого бара «сгорают»
        tf_ltf_min = TF_MINUTES[p.tf_ltf]
        if not self._session_ok(now + pd.Timedelta(minutes=tf_ltf_min)):
            p.scanned_until = now
            return None

        idx = ltf_df.index
        start_pos = int(idx.searchsorted(p.pierce_time))
        if start_pos >= len(idx):
            return None
        scan_from = start_pos
        if p.scanned_until is not None:
            scan_from = max(scan_from, int(idx.searchsorted(p.scanned_until, side="right")))
        p.scanned_until = now

        a_ltf = atr_value(ltf_df.tail(120))
        min_size = a_ltf * s.ifvg_min_size_atr
        fvg_from = max(2, start_pos - s.ifvg_search_back)
        res = find_first_inversion(ltf_df, p.direction, fvg_from, scan_from, min_size)
        if res is None:
            return None
        fvg, inv_j = res

        entry = float(ltf_df["close"].iloc[-1])
        seg = ltf_df.iloc[start_pos:]
        seg_ext = float(seg["low"].min()) if p.direction == "buy" \
            else float(seg["high"].max())
        sig = self._finalize(symbol, model_id, p.direction, p.extreme, seg_ext,
                             entry, p.pool_low, p.pool_high, p.atr_main,
                             p.smt, p.quasimodo, p.trend, now)
        self.pending.pop(key, None)
        if sig:
            log.info("[%s %s] СИГНАЛ %s: entry=%.5f sl=%.5f tp=%.5f rr=%.2f "
                     "(iFVG %s, бар %d)", symbol, model_id, sig.direction.upper(),
                     sig.entry, sig.sl, sig.tp, sig.rr, fvg.kind, inv_j)
        return sig

    # ------------------------- построение сигнала -------------------------
    def _finalize(self, symbol: str, model_id: str, direction: str,
                  sweep_extreme: float, seg_extreme: float, entry: float,
                  pool_low: float, pool_high: float, atr_main: float,
                  smt: Optional[bool], qm: bool, trend: str,
                  now: pd.Timestamp) -> Optional[Signal]:
        s = self.s
        buf = atr_main * s.sl_buffer_atr
        if direction == "buy":
            base = min(sweep_extreme, seg_extreme)
            sl = base - buf
            risk = entry - sl
            if risk <= 0 or entry >= pool_high:
                return None
            tp = pool_high if (pool_high - entry) / risk >= s.min_rr \
                else entry + s.rr_fallback * risk
        else:
            base = max(sweep_extreme, seg_extreme)
            sl = base + buf
            risk = sl - entry
            if risk <= 0 or entry <= pool_low:
                return None
            tp = pool_low if (entry - pool_low) / risk >= s.min_rr \
                else entry - s.rr_fallback * risk

        rr = abs(tp - entry) / risk
        tags = []
        if smt:
            tags.append("SMT")
        if qm:
            tags.append("QM")
        comment = f"{'+'.join(tags) if tags else 'SWEEP'} {model_id}"
        return Signal(symbol, model_id, direction, now, entry, float(sl), float(tp),
                      rr, smt, qm, trend, comment)
