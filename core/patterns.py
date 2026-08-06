"""Паттерны: двухсвечной свип ликвидности, FVG / inversion FVG, SMT-дивергенция."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .structure import Pools, find_swings


# ----------------------------- СВИП -----------------------------

@dataclass
class Sweep:
    side: str                # 'low' | 'high'
    level: float             # цена пула ликвидности
    extreme: float           # экстремум хвоста (мин/макс двух свечей)
    pierce_idx: int          # индекс свечи-прокола
    react_idx: int           # индекс свечи-реакции
    pierce_time: pd.Timestamp
    time: pd.Timestamp       # время (открытия) реакционной свечи


def _first_close_through(close: np.ndarray, level: float, side: str, start: int) -> Optional[int]:
    """Первый закреп телом за уровнем (закрытие за уровень) — уровень 'пробит'."""
    for i in range(start, len(close)):
        if side == "low" and close[i] < level:
            return i
        if side == "high" and close[i] > level:
            return i
    return None


def find_recent_sweep(df: pd.DataFrame, pools: Pools, max_age: int = 3,
                      trend: str = "range", with_trend_only: bool = False) -> Optional[Sweep]:
    """Свип внешней ликвидности строго из двух свечей:
      1) свеча-прокол: хвост за уровнем, но БЕЗ закрепа (закрытие остаётся по нужную сторону);
      2) свеча-реакция: тело в обратную сторону с закрытием от уровня.
    Если до предполагаемого свипа уже был закреп телом за уровнем — свип не засчитывается.
    Возвращает самый свежий свип (реакция в пределах последних max_age баров)."""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    if n < 5:
        return None

    best: Optional[Sweep] = None
    for side, sw in (("low", pools.low), ("high", pools.high)):
        if sw is None:
            continue
        level = sw.price
        ct = _first_close_through(c, level, side, sw.idx + 1)
        lo_i = max(sw.idx + 1, n - max_age - 1)
        for i in range(lo_i, n - 1):
            if ct is not None and i >= ct:
                break
            j = i + 1
            if side == "low":
                pierced = l[i] < level and c[i] >= level
                react = c[j] > o[j] and c[j] > level
                if pierced and react:
                    ev = Sweep("low", level, float(min(l[i], l[j])), i, j,
                               df.index[i], df.index[j])
                    if best is None or ev.react_idx > best.react_idx:
                        best = ev
            else:
                pierced = h[i] > level and c[i] <= level
                react = c[j] < o[j] and c[j] < level
                if pierced and react:
                    ev = Sweep("high", level, float(max(h[i], h[j])), i, j,
                               df.index[i], df.index[j])
                    if best is None or ev.react_idx > best.react_idx:
                        best = ev

    if best is not None and with_trend_only and trend in ("up", "down"):
        want = "low" if trend == "up" else "high"
        if best.side != want:
            return None
    return best


# ----------------------------- FVG / iFVG -----------------------------

@dataclass
class FVG:
    kind: str      # 'bull' | 'bear' — направление имбаланса
    top: float
    bottom: float
    idx: int       # индекс третьей свечи (создание гэпа)


def find_fvgs(df: pd.DataFrame, start: int, end: int, min_size: float = 0.0) -> List[FVG]:
    """FVG по трём свечам: bull — low[i] > high[i-2]; bear — high[i] < low[i-2].
    `end` не включается."""
    h = df["high"].values
    l = df["low"].values
    out: List[FVG] = []
    for i in range(max(2, start), min(end, len(df))):
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_size:
            out.append(FVG("bull", top=float(l[i]), bottom=float(h[i - 2]), idx=i))
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_size:
            out.append(FVG("bear", top=float(l[i - 2]), bottom=float(h[i]), idx=i))
    return out


def find_first_inversion(df: pd.DataFrame, direction: str, fvg_from: int,
                         scan_from: int, min_size: float = 0.0) -> Optional[Tuple[FVG, int]]:
    """Инверсия FVG = полное закрепление ТЕЛОМ против имбаланса (по SMC):
      для покупки  — медвежий FVG, закрытие ВЫШЕ его верхней границы (гэп перекрыт телом);
      для продажи  — бычий FVG,   закрытие НИЖЕ его нижней границы.
    Учитывается только ПЕРВАЯ инверсия каждого FVG; возвращается самая ранняя
    инверсия с индексом >= scan_from (чтобы не сигналить повторно)."""
    c = df["close"].values
    n = len(df)
    kind = "bear" if direction == "buy" else "bull"
    fvgs = [f for f in find_fvgs(df, fvg_from, n, min_size) if f.kind == kind]
    best: Optional[Tuple[FVG, int]] = None
    for f in fvgs:
        for j in range(f.idx + 1, n):
            inverted = c[j] > f.top if direction == "buy" else c[j] < f.bottom
            if inverted:
                if j >= scan_from and (best is None or j < best[1]):
                    best = (f, j)
                break
    return best


# ----------------------------- SMT -----------------------------

def invert_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Инвертирование серии (iDXY): экстремумы меняются местами со сменой знака,
    так что свип минимума у инвертированной серии соответствует свипу максимума
    у исходной. Для логики свингов/свипов важны только относительные сравнения."""
    out = pd.DataFrame(index=df.index)
    out["open"] = -df["open"]
    out["close"] = -df["close"]
    out["high"] = -df["low"]
    out["low"] = -df["high"]
    return out


def smt_check(corr_df: Optional[pd.DataFrame], sweep: Sweep, level_time: pd.Timestamp,
              swing_left: int = 2, swing_right: int = 2,
              max_ref_dist: Optional[pd.Timedelta] = None) -> Optional[bool]:
    """SMT-дивергенция по снятию ликвидности:
    основной актив СНЯЛ свой экстремум (sweep), а коррелят соответствующий
    экстремум НЕ снял -> True (дивергенция, сильный сигнал).
    Коррелят должен быть заранее инвертирован для обратных пар (iDXY).
    None — проверить невозможно (нет данных / нет опорного свинга)."""
    if corr_df is None or len(corr_df) < 30:
        return None
    csw = find_swings(corr_df, swing_left, swing_right)
    kind = "L" if sweep.side == "low" else "H"
    cands = [x for x in csw if x.kind == kind and x.time <= sweep.pierce_time]
    if not cands:
        return None
    # опорный экстремум коррелята — ближайший по времени к формированию уровня основного
    ref = min(cands, key=lambda x: abs((x.time - level_time).total_seconds()))
    if max_ref_dist is not None and abs(ref.time - level_time) > max_ref_dist:
        return None
    seg = corr_df[(corr_df.index > ref.time) & (corr_df.index <= sweep.time)]
    if seg.empty:
        return None
    if sweep.side == "low":
        swept = float(seg["low"].min()) < ref.price
    else:
        swept = float(seg["high"].max()) > ref.price
    return not swept
