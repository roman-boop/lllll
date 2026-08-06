"""
Конфигурация торговой системы SMT / Quasimodo (MT5).
Все ключевые параметры логики, сессии, лимитов и риска собраны здесь.
"""
from dataclasses import dataclass

MAGIC = 557001  # magic number ордеров системы

# Основные инструменты и их SMT-корреляты.
# invert=True -> коррелят инвертируется перед сравнением (iDXY = инвертированный индекс доллара)
SMT_MAP = {
    "XAUUSD": {"corr": "XAGUSD", "invert": False},
    "XAGUSD": {"corr": "XAUUSD", "invert": False},
    "EURUSD": {"corr": "DXY", "invert": True},
}

# У брокеров индекс доллара / металлы называются по-разному — перебираем алиасы
DXY_ALIASES = ["DXY", "USDX", "USDIDX", "USIDX", "DX", "USDIndex", "USDOLLAR", "USDollar"]
SILVER_ALIASES = ["XAGUSD", "SILVER", "XAGUSDm", "XAGUSD."]
GOLD_ALIASES = ["XAUUSD", "GOLD", "XAUUSDm", "XAUUSD."]

# ТФ-модели: main — поиск структуры/свипов, ltf — подтверждение через inversion FVG.
# enabled=False отключает модель без удаления из списка.
TF_MODELS = [
    {"main": "M15", "ltf": "M5", "enabled": True},
    {"main": "H1",  "ltf": "M5", "enabled": True},
    {"main": "M5",  "ltf": "M3", "enabled": True},
]

TF_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10, "M12": 12,
    "M15": 15, "M20": 20, "M30": 30, "H1": 60, "H2": 120, "H4": 240, "D1": 1440,
}


@dataclass
class Settings:
    # =========== ПЕРЕКЛЮЧАТЕЛИ СТРАТЕГИЙ ===========
    use_smt: bool = True          # False: SMT полностью выключена (торгуем без дивергенций)
    use_quasimodo: bool = True    # False: Quasimodo не используется (ни конфлюэнс, ни замена SMT)
    require_qm: bool = False      # True: входы ТОЛЬКО при Quasimodo-конфлюэнсе
    require_ifvg: bool = True     # False: вход сразу по свипу (без iFVG-подтверждения на LTF)
    with_trend_only: bool = False # True: только по тренду (свип минимума в аптренде / максимума в даунтренде)

    # --- рыночная структура ---
    swing_left: int = 2            # баров слева от фрактального свинга
    swing_right: int = 2           # баров справа (лаг подтверждения свинга)
    hist_main_bars: int = 500      # глубина анализа main TF
    hist_ltf_bars: int = 900       # глубина анализа LTF
    eq_tolerance_atr: float = 0.05 # допуск равенства экстремумов (EQH/EQL), доля ATR

    # --- свип внешней ликвидности ---
    max_sweep_age: int = 3         # реакционная свеча не старше N закрытых main-баров
    confirm_window_main: int = 8   # окно ожидания iFVG-подтверждения, в барах main TF

    # --- inversion FVG (LTF) ---
    ifvg_min_size_atr: float = 0.03  # мин. размер FVG (доля ATR LTF) — фильтр шума
    ifvg_search_back: int = 40       # на сколько LTF-баров назад от свипа искать FVG

    # --- SMT / Quasimodo ---
    # Режим действует, только если use_smt=True:
    # 'prefer'   — SMT лишь помечает сделку (не фильтрует)
    # 'require'  — вход только при подтверждённой SMT-дивергенции (или QM, см. ниже)
    # 'require_if_available' — как require, но если данных коррелята нет — вход разрешён
    smt_mode: str = "require_if_available"
    qm_substitutes_smt: bool = True  # конфлюэнс с Quasimodo может заменить SMT
    smt_ref_window: int = 60         # окно (в main-барах) поиска опорного свинга коррелята

    # =========== ТОРГОВАЯ СЕССИЯ (Нью-Йорк) ===========
    session_enabled: bool = True
    session_tz: str = "America/New_York"  # таймзона окна (DST учитывается автоматически)
    session_start: str = "02:00"          # включительно
    session_end: str = "10:00"            # исключительно; окно через полночь тоже поддерживается
    # Смещение времени БАРОВ данных от UTC (время сервера брокера, обычно +2 или +3).
    # Используется в бэктесте; в realtime сессия проверяется по системным часам.
    bars_utc_offset_hours: float = 0.0

    # =========== ДНЕВНОЙ ЛИМИТ УБЫТКА ===========
    max_daily_loss_pct: float = 0.03      # 0.03 = 3% от эквити на начало дня; 0 = выключено
    max_total_loss_pct: float = 0.0       # ОБЩИЙ стоп от стартового баланса (0 = выкл);
                                          # для пропа с лимитом 10% ставьте 0.07–0.08
    daily_loss_close_positions: bool = False  # True: при срабатывании закрыть и открытые позиции системы
    # День считается по session_tz; после срабатывания новые входы блокируются до следующего дня.

    # --- риск / цели ---
    risk_per_trade: float = 0.005  # доля эквити на сделку (0.005 = 0.5%)
    min_rr: float = 2.0            # мин. RR до противоположного пула, иначе цель по RR
    rr_fallback: float = 2.5       # RR-цель, если пул слишком близко
    sl_buffer_atr: float = 0.1     # буфер SL за экстремум свипа, доля ATR main
    breakeven_at_r: float = 0.0    # 0 = выкл; напр. 1.0 = перенос SL в БУ при +1R (realtime)
    one_position_per_symbol: bool = True

    # --- realtime ---
    poll_seconds: float = 3.0
    deviation_points: int = 30

    # --- backtest ---
    spread: float = 0.0            # спред в ценовых единицах (ухудшает результат каждой сделки)
    conservative_fills: bool = True  # если SL и TP в одном баре — считаем SL
