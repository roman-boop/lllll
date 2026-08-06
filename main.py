"""
Realtime SMT / Quasimodo система для MetaTrader 5 (металлы и форекс).

Логика:
  1. На main TF (M15 / H1 / M5) строится структура (HH/HL/LL/LH), определяется тренд
     и два пула внешней ликвидности (в аптренде — последний минимум и максимум после него).
  2. Ловится двухсвечной свип пула: прокол хвостом БЕЗ закрепа + свеча-реакция.
     Свип минимума -> лонг, свип максимума -> шорт.
  3. Проверяется SMT-дивергенция по снятию ликвидности с коррелятом
     (XAUUSD <-> XAGUSD, EURUSD <-> iDXY) и конфлюэнс с Quasimodo-уровнем.
  4. Вход — после инверсии FVG на LTF (полное закрепление телом против имбаланса):
     модели 15m+5m, 1h+5m, 5m+3m. Режим --sweep-only входит сразу по свипу.
  5. SL за экстремум свипа (+буфер), TP — противоположный пул ликвидности (или RR-цель).

Ограничения торговли:
  * входы только в торговую сессию 02:00–10:00 по Нью-Йорку (config / CLI);
  * дневной лимит убытка 3% от эквити на начало дня — после срабатывания новые
    входы блокируются до следующего дня (config / CLI).

Запуск (Windows, установленный терминал MT5):
  python main.py --symbols XAUUSD EURUSD --dry-run
  python main.py --symbols XAUUSD --sweep-only --no-smt --risk 0.005
  python main.py --login 123456 --password *** --server Broker-Demo
"""
import argparse
import logging
import sys
import time

import pandas as pd

from config import (Settings, TF_MODELS, SMT_MAP, DXY_ALIASES, SILVER_ALIASES,
                    GOLD_ALIASES, MAGIC)
from core.mt5_client import MT5Client
from core.engine import SignalEngine
from core.patterns import invert_ohlc
from core.session import local_day
from execution.trader import Trader


def setup_logging():
    fmt = "%(asctime)s %(levelname)-7s %(name)-7s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("smt_system.log", encoding="utf-8")],
    )


def parse_args():
    p = argparse.ArgumentParser(description="SMT/Quasimodo realtime система (MT5)")
    p.add_argument("--symbols", nargs="+", default=["XAUUSD", "EURUSD"])
    p.add_argument("--models", nargs="+",
                   help="какие ТФ-модели включить, напр.: M15/M5 H1/M5 (по умолчанию — enabled из config)")
    p.add_argument("--login")
    p.add_argument("--password")
    p.add_argument("--server")
    p.add_argument("--path", help="путь к terminal64.exe (если терминалов несколько)")
    p.add_argument("--risk", type=float, help="риск на сделку, доля эквити (0.005 = 0.5%%)")
    # переключатели стратегий
    p.add_argument("--no-smt", action="store_true", help="выключить SMT (торговля без дивергенций)")
    p.add_argument("--no-qm", action="store_true", help="выключить Quasimodo")
    p.add_argument("--require-qm", action="store_true",
                   help="входы только при Quasimodo-конфлюэнсе")
    p.add_argument("--no-qm-substitute", action="store_true",
                   help="QM не заменяет SMT (строгий конфлюэнс SMT+QM)")
    p.add_argument("--sweep-only", action="store_true",
                   help="входить сразу по свипу, без iFVG-подтверждения на LTF")
    p.add_argument("--smt-mode", choices=["prefer", "require", "require_if_available"])
    p.add_argument("--with-trend-only", action="store_true",
                   help="торговать только по тренду (свип минимума в аптренде и т.п.)")
    # сессия и лимиты
    p.add_argument("--no-session", action="store_true", help="снять ограничение торговой сессии")
    p.add_argument("--session-start", help="начало окна HH:MM по Нью-Йорку (по умолчанию 02:00)")
    p.add_argument("--session-end", help="конец окна HH:MM по Нью-Йорку (по умолчанию 10:00)")
    p.add_argument("--max-daily-loss", type=float,
                   help="дневной лимит убытка, доля эквити (0.03 = 3%%; 0 = выкл)")
    p.add_argument("--max-total-loss", type=float,
                   help="ОБЩИЙ лимит убытка от стартового баланса (0.08 = 8%%; 0 = выкл)")
    p.add_argument("--initial-balance", type=float,
                   help="стартовый баланс проп-счёта — база общего лимита (переживает перезапуски)")
    p.add_argument("--close-on-limit", action="store_true",
                   help="при срабатывании дневного лимита также закрывать открытые позиции")
    p.add_argument("--breakeven-r", type=float, help="перенос SL в БУ при +N R (0 = выкл)")
    p.add_argument("--dry-run", action="store_true", help="только сигналы, без ордеров")
    return p.parse_args()


def apply_args(s: Settings, args) -> Settings:
    if args.risk is not None:
        s.risk_per_trade = args.risk
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
    if args.max_daily_loss is not None:
        s.max_daily_loss_pct = args.max_daily_loss
    if args.max_total_loss is not None:
        s.max_total_loss_pct = args.max_total_loss
    if args.close_on_limit:
        s.daily_loss_close_positions = True
    if args.breakeven_r is not None:
        s.breakeven_at_r = args.breakeven_r
    return s


def select_models(args):
    models = [m for m in TF_MODELS if m.get("enabled", True)]
    if args.models:
        want = {x.upper() for x in args.models}
        models = [m for m in TF_MODELS
                  if f"{m['main']}/{m['ltf']}".upper() in want]
    if not models:
        logging.error("Не выбрано ни одной ТФ-модели (проверьте --models / config.TF_MODELS)")
        sys.exit(1)
    return models


def resolve_correlates(client: MT5Client, symbols, use_smt: bool):
    """Подбираем тикеры коррелятов у конкретного брокера (алиасы DXY / металлов)."""
    corr = {}
    if not use_smt:
        return {sym: None for sym in symbols}
    alias_map = {"DXY": DXY_ALIASES, "XAGUSD": SILVER_ALIASES, "XAUUSD": GOLD_ALIASES}
    for sym in symbols:
        cfg = None
        for base, c in SMT_MAP.items():
            if sym.upper().startswith(base):
                cfg = c
                break
        if not cfg:
            corr[sym] = None
            logging.warning("Для %s SMT-коррелят не настроен (config.SMT_MAP)", sym)
            continue
        name = client.resolve_symbol(alias_map.get(cfg["corr"], [cfg["corr"]]))
        if name is None:
            logging.warning("Коррелят для %s не найден у брокера (%s) — SMT будет недоступен",
                            sym, cfg["corr"])
            corr[sym] = None
        else:
            corr[sym] = {"symbol": name, "invert": cfg["invert"]}
            logging.info("SMT-коррелят для %s: %s%s", sym, name,
                         " (инвертирован, iDXY)" if cfg["invert"] else "")
    return corr


def main():
    setup_logging()
    args = parse_args()
    s = apply_args(Settings(), args)
    models = select_models(args)

    client = MT5Client()
    attempt = 0
    while not client.connect(args.login, args.password, args.server, args.path):
        attempt += 1
        logging.warning("Терминал MT5 недоступен (попытка %d) — повтор через 30 с. "
                        "Проверьте, что терминал запущен и выполнен вход.", attempt)
        time.sleep(30)

    symbols = [client.resolve_symbol([x]) or x for x in args.symbols]
    corr_map = resolve_correlates(client, symbols, s.use_smt)
    engine = SignalEngine(s, now_fn=lambda: pd.Timestamp.now(tz="UTC"))
    trader = Trader(client, s, dry_run=args.dry_run)

    last_main = {}   # (sym, model_id) -> время последнего обработанного main-бара
    last_ltf = {}

    # состояние дневного лимита убытка
    day_key = None
    day_start_eq = 0.0
    locked = False
    # общий лимит убытка (проп-стоп) и вотчдог связи
    total_baseline = float(args.initial_balance or 0.0)
    total_locked = False
    fail_streak = 0

    logging.info("Старт. Символы: %s | Модели: %s | SMT=%s QM=%s iFVG=%s | "
                 "Сессия: %s %s-%s (%s) | Дневной лимит: %s | dry_run=%s",
                 symbols, [f"{m['main']}/{m['ltf']}" for m in models],
                 s.use_smt and s.smt_mode, s.use_quasimodo, s.require_ifvg,
                 "вкл" if s.session_enabled else "ВЫКЛ",
                 s.session_start, s.session_end, s.session_tz,
                 f"{s.max_daily_loss_pct:.1%}" if s.max_daily_loss_pct > 0 else "выкл",
                 args.dry_run)

    def handle_signal(sig):
        nonlocal locked
        if sig is None:
            return
        if locked:
            logging.info("[%s] сигнал %s пропущен: достигнут дневной лимит убытка",
                         sig.symbol, sig.direction.upper())
            return
        trader.execute(sig)

    try:
        while True:
            eq_now = client.equity()  # 0.0 = терминал не отвечает

            # ---- общий лимит убытка от стартового баланса (проп-стоп) ----
            if s.max_total_loss_pct > 0 and eq_now > 0:
                if total_baseline <= 0:
                    total_baseline = eq_now
                    logging.info("База общего лимита убытка: %.2f (лимит %.1f%%). "
                                 "Для проп-счёта задавайте её явно: --initial-balance",
                                 total_baseline, s.max_total_loss_pct * 100)
                if not total_locked and eq_now <= total_baseline * (1.0 - s.max_total_loss_pct):
                    total_locked = True
                    logging.critical(
                        "ОБЩИЙ ЛИМИТ УБЫТКА %.1f%% ДОСТИГНУТ (эквити %.2f, база %.2f) — "
                        "торговля ОСТАНОВЛЕНА, позиции закрываются. Перезапуск только вручную.",
                        s.max_total_loss_pct * 100, eq_now, total_baseline)
                    if not args.dry_run:
                        client.close_all(MAGIC, "total-loss-stop")
            if total_locked:
                time.sleep(s.poll_seconds)
                continue

            # ---- дневной лимит убытка (день по session_tz) ----
            if s.max_daily_loss_pct > 0 and eq_now > 0:
                today = local_day(pd.Timestamp.now(tz="UTC"), s)
                if today != day_key:
                    day_key = today
                    day_start_eq = eq_now
                    locked = False
                    logging.info("Новый торговый день %s (%s), стартовый эквити %.2f",
                                 today, s.session_tz, day_start_eq)
                if not locked and day_start_eq > 0:
                    if eq_now <= day_start_eq * (1.0 - s.max_daily_loss_pct):
                        locked = True
                        logging.warning(
                            "ДНЕВНОЙ ЛИМИТ УБЫТКА %.1f%% ДОСТИГНУТ (эквити %.2f, "
                            "старт дня %.2f) — новые входы заблокированы до следующего дня",
                            s.max_daily_loss_pct * 100, eq_now, day_start_eq)
                        if s.daily_loss_close_positions and not args.dry_run:
                            client.close_all(MAGIC)

            got_data = False
            for sym in symbols:
                for model in models:
                    mid = f"{model['main']}/{model['ltf']}"
                    key = (sym, mid)

                    # ---- новый закрытый бар main TF: структура/свип/SMT/QM ----
                    main_df = client.rates(sym, model["main"], s.hist_main_bars)
                    if main_df is None or len(main_df) < 60:
                        continue
                    got_data = True
                    t_main = main_df.index[-1]
                    if last_main.get(key) != t_main:
                        corr_df = None
                        cc = corr_map.get(sym)
                        if cc:
                            corr_df = client.rates(cc["symbol"], model["main"],
                                                   s.hist_main_bars)
                            if corr_df is not None and cc["invert"]:
                                corr_df = invert_ohlc(corr_df)
                        sig = engine.on_main_bar(sym, mid, model["main"],
                                                 model["ltf"], main_df, corr_df)
                        last_main[key] = t_main
                        handle_signal(sig)  # режим --sweep-only

                    # ---- есть pending-сетап: ждём iFVG на LTF ----
                    if key in engine.pending:
                        ltf_df = client.rates(sym, model["ltf"], s.hist_ltf_bars)
                        if ltf_df is None or len(ltf_df) < 10:
                            continue
                        t_ltf = ltf_df.index[-1]
                        if last_ltf.get(key) != t_ltf:
                            last_ltf[key] = t_ltf
                            handle_signal(engine.on_ltf_bar(sym, mid, ltf_df))

            # ---- вотчдог связи с терминалом ----
            if got_data or eq_now > 0:
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= 20:  # ~1 минута без данных
                    logging.warning("Связь с терминалом потеряна (~%d с) — переподключение...",
                                    int(fail_streak * s.poll_seconds))
                    client.shutdown()
                    if client.connect(args.login, args.password, args.server, args.path):
                        logging.info("Переподключение к терминалу выполнено")
                    fail_streak = 0

            trader.manage_breakeven()
            time.sleep(s.poll_seconds)
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
