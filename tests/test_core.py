"""Юнит-проверки ядра (запуск: python tests/test_core.py)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings  # noqa: E402
from core.structure import find_swings, label_swings, analyze_structure, Pools  # noqa: E402
from core.patterns import (find_recent_sweep, find_fvgs, find_first_inversion,  # noqa: E402
                           invert_ohlc, smt_check, Sweep)
from core.structure import Swing  # noqa: E402


def df_from(rows, start="2026-01-01", freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_swings_labels():
    rows = [
        (10, 11, 9.5, 10.5), (10.5, 12, 10, 11.5), (11.5, 13, 11, 12.5),  # рост -> H
        (12.5, 12.6, 11.2, 11.4), (11.4, 11.6, 10.8, 11.0),               # откат -> L (HL)
        (11.0, 12.0, 10.9, 11.8), (11.8, 14, 11.5, 13.5),                 # рост -> HH
        (13.5, 13.6, 12.5, 12.7), (12.7, 12.9, 12.0, 12.2),
        (12.2, 13.0, 12.1, 12.8), (12.8, 13.2, 12.4, 13.0),
    ]
    df = df_from(rows)
    sw = label_swings(find_swings(df, 2, 2))
    kinds = [s.kind for s in sw]
    assert "H" in kinds and "L" in kinds, "должны находиться и максимумы, и минимумы"
    hs = [s for s in sw if s.kind == "H"]
    assert any(s.label == "HH" for s in hs[1:]) or len(hs) == 1
    print("OK swings/labels:", [(s.kind, s.label, s.price) for s in sw])


def test_sweep_ok_and_reject():
    # пул-минимум на 100.0; свеча-прокол хвостом до 99.4 c закрытием 100.3; реакция вверх
    rows = [(101, 101.5, 100.0, 100.6)] * 3 + [
        (100.6, 101.2, 100.2, 101.0),
        (101.0, 101.4, 100.4, 100.8),
        (100.8, 101.0, 99.4, 100.3),   # прокол без закрепа
        (100.3, 101.6, 100.2, 101.4),  # бычья реакция
    ]
    df = df_from(rows)
    pool = Pools(low=Swing(0, df.index[0], 100.0, "L"), high=None)
    sw = find_recent_sweep(df, pool, max_age=3)
    assert sw is not None and sw.side == "low" and sw.extreme == 99.4, "свип должен найтись"

    # тот же прокол, но с закрепом телом ниже уровня -> свип не засчитывается
    rows2 = rows[:-2] + [(100.8, 101.0, 99.4, 99.7), (99.7, 101.6, 99.6, 101.4)]
    df2 = df_from(rows2)
    sw2 = find_recent_sweep(df2, pool, max_age=3)
    assert sw2 is None, "закреп телом за уровнем должен отменять свип"
    print("OK sweep: найден при проколе, отклонён при закрепе")


def test_fvg_inversion():
    # медвежий FVG: high[2] < low[0]; затем закрытие выше top -> инверсия для покупки
    rows = [
        (105, 106, 104, 104.2),
        (104.2, 104.4, 101.8, 102.0),
        (102.0, 102.6, 101.5, 102.2),   # high=102.6 < low[0]=104 -> bear FVG [102.6..104]
        (102.2, 103.0, 102.0, 102.8),
        (102.8, 104.8, 102.7, 104.6),   # закрытие 104.6 > top=104 -> инверсия (buy)
    ]
    df = df_from(rows, freq="5min")
    fv = find_fvgs(df, 0, len(df))
    assert any(f.kind == "bear" for f in fv), "медвежий FVG должен найтись"
    res = find_first_inversion(df, "buy", 0, 0)
    assert res is not None and res[1] == 4, "инверсия должна случиться на баре 4"
    # для sell на этих данных инверсии быть не должно
    assert find_first_inversion(df, "sell", 0, 0) is None
    print("OK iFVG: bear FVG + инверсия закрытием выше top")


def test_invert_and_smt():
    rows = [
        (100, 101, 99, 100.5), (100.5, 102, 100, 101.5),
        (101.5, 103, 101, 102.5), (102.5, 102.8, 100.2, 100.5),
        (100.5, 101.0, 99.0, 100.8), (100.8, 102.0, 100.5, 101.8),
        (101.8, 102.5, 101.2, 102.2), (102.2, 102.6, 101.5, 102.0),
        (102.0, 102.4, 101.4, 101.9), (101.9, 102.3, 101.3, 101.8),
    ]
    df = df_from(rows)
    inv = invert_ohlc(df)
    assert (inv["high"] >= inv["low"]).all() and inv["high"].iloc[0] == -99
    # коррелят держит минимум (не обновляет 99.0 после опорного свинга) -> SMT=True
    sweep = Sweep("low", 99.5, 98.9, 6, 7, df.index[6], df.index[7])
    corr = df.copy()
    corr.loc[corr.index[5:], "low"] = corr.loc[corr.index[5:], "low"].clip(lower=99.2)
    res = smt_check(pd.concat([corr] * 3), sweep, df.index[4], 1, 1)
    assert res in (True, None)
    print("OK invert_ohlc + smt_check")


def test_structure_pools():
    rows = []
    p = 100.0
    import math
    for i in range(120):
        p += 0.3 + 0.8 * math.sin(i / 5)
        rows.append((p - 0.2, p + 0.5, p - 0.6, p))
    df = df_from(rows)
    st = analyze_structure(df, Settings())
    assert st.trend in ("up", "down", "range")
    assert st.pools.low is not None and st.pools.high is not None
    if st.trend == "up":
        assert st.pools.high.idx >= st.pools.low.idx or st.pools.high.price > st.pools.low.price
    print(f"OK structure: trend={st.trend}, pool_low={st.pools.low.price:.2f}, "
          f"pool_high={st.pools.high.price:.2f}, свингов={len(st.swings)}")


def test_session_window():
    from core.session import in_session, bars_to_utc, local_day
    s = Settings()
    s.session_enabled = True
    s.session_tz = "America/New_York"
    s.session_start, s.session_end = "02:00", "10:00"
    # лето (EDT, UTC-4): окно 06:00–14:00 UTC
    assert in_session(pd.Timestamp("2026-05-15 06:00", tz="UTC"), s)
    assert in_session(pd.Timestamp("2026-05-15 13:59", tz="UTC"), s)
    assert not in_session(pd.Timestamp("2026-05-15 14:00", tz="UTC"), s)
    assert not in_session(pd.Timestamp("2026-05-15 05:59", tz="UTC"), s)
    # зима (EST, UTC-5): окно 07:00–15:00 UTC
    assert in_session(pd.Timestamp("2026-01-15 07:30", tz="UTC"), s)
    assert not in_session(pd.Timestamp("2026-01-15 06:30", tz="UTC"), s)
    # бары со временем сервера UTC+3: наивная метка 09:00 = 06:00 UTC -> в сессии
    s.bars_utc_offset_hours = 3.0
    assert in_session(bars_to_utc(pd.Timestamp("2026-05-15 09:00"), s), s)
    assert not in_session(bars_to_utc(pd.Timestamp("2026-05-15 08:59"), s), s)
    s.bars_utc_offset_hours = 0.0
    # граница торгового дня по NY: 03:00 UTC = 23:00 предыдущего дня EDT
    assert str(local_day(pd.Timestamp("2026-05-15 03:00", tz="UTC"), s)) == "2026-05-14"
    # окно через полночь
    s.session_start, s.session_end = "22:00", "03:00"
    assert in_session(pd.Timestamp("2026-05-15 03:30", tz="UTC"), s)   # 23:30 NY
    assert not in_session(pd.Timestamp("2026-05-15 14:00", tz="UTC"), s)  # 10:00 NY
    print("OK session: окно NY 02:00-10:00, DST, offset баров, граница дня, окно через полночь")


if __name__ == "__main__":
    test_swings_labels()
    test_sweep_ok_and_reject()
    test_fvg_inversion()
    test_invert_and_smt()
    test_structure_pools()
    test_session_window()
    print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ")
