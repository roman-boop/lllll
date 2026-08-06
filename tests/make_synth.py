"""Генерация синтетических коррелированных данных для проверки системы.

Строится минутный ряд (random walk с трендовыми режимами и врезанными
V-разворотами со свипом), затем ресемплится в M3/M5/M15/H1.
Коррелят повторяет базовый ряд с шумом, но в местах части свипов
экстремум НЕ обновляет (готовая SMT-дивергенция).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import resample_ohlc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build(seed: int = 7, days: int = 30):
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    t = pd.date_range("2026-05-01", periods=n, freq="1min")

    drift = np.zeros(n)
    seg = n // 10
    for k in range(10):
        drift[k * seg:(k + 1) * seg] = rng.choice([-0.004, 0.0, 0.004, 0.006, -0.006])
    noise = rng.normal(0, 0.06, n)
    price = 2400.0 + np.cumsum(drift + noise)

    # врезаем V-развороты (свип-паттерны): резкий заброс и возврат
    sweep_marks = []
    for pos in range(2000, n - 2000, 2500):
        side = rng.choice(["low", "high"])
        depth = rng.uniform(0.8, 1.8)
        w = 25
        shape = np.concatenate([np.linspace(0, depth, w // 2),
                                np.linspace(depth, -0.2 * depth, w - w // 2)])
        price[pos:pos + w] += -shape if side == "low" else shape
        sweep_marks.append((pos, side))

    close = price
    o = np.empty(n)
    o[0] = close[0]
    o[1:] = close[:-1]
    spread_hl = np.abs(rng.normal(0, 0.05, n)) + 0.02
    high = np.maximum(o, close) + spread_hl
    low = np.minimum(o, close) - spread_hl
    base = pd.DataFrame({"open": o, "high": high, "low": low, "close": close}, index=t)

    # коррелят: копия с шумом; у половины свипов экстремум гасим (SMT-дивергенция)
    c = price.copy() * 0.5 + rng.normal(0, 0.05, n).cumsum() * 0.15 + 600
    for i, (pos, side) in enumerate(sweep_marks):
        if i % 2 == 0:
            w = 25
            seg_sl = slice(pos, pos + w)
            local = c[seg_sl]
            c[seg_sl] = np.clip(local, local[0] - 0.15, local[0] + 0.15)
    co = np.empty(n)
    co[0] = c[0]
    co[1:] = c[:-1]
    chl = np.abs(rng.normal(0, 0.03, n)) + 0.01
    corr = pd.DataFrame({"open": co, "high": np.maximum(co, c) + chl,
                         "low": np.minimum(co, c) - chl, "close": c}, index=t)

    os.makedirs(OUT, exist_ok=True)
    for name, df in (("main", base), ("corr", corr)):
        for tf, m in (("m3", 3), ("m5", 5), ("m15", 15), ("h1", 60)):
            r = resample_ohlc(df, m)
            r.insert(0, "time", r.index)
            r.to_csv(os.path.join(OUT, f"{name}_{tf}.csv"), index=False)
    print(f"OK: синтетика сохранена в {OUT} ({days} дней, {len(sweep_marks)} свип-паттернов)")


if __name__ == "__main__":
    build()
