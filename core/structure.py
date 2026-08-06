"""Рыночная структура: свинги (HH/HL/LL/LH), тренд, пулы внешней ликвидности, Quasimodo.

Свинги — фрактальные, подтверждаются только при наличии `right` закрытых баров справа,
поэтому при последовательном проходе (бэктест) заглядывания в будущее нет.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Swing:
    idx: int
    time: pd.Timestamp
    price: float
    kind: str            # 'H' | 'L'
    label: str = ""      # HH/LH/EQH | HL/LL/EQL


@dataclass
class Pools:
    """Внешняя ликвидность: два последних по тренду пула."""
    low: Optional[Swing]     # sell-side (минимум)
    high: Optional[Swing]    # buy-side (максимум)


@dataclass
class QMLevel:
    direction: str           # 'sell' | 'buy'
    qml: float               # уровень левого плеча (QM-level)
    head: float              # экстремум «головы» (снятие ликвидности)
    time: pd.Timestamp


@dataclass
class Structure:
    swings: List[Swing]
    trend: str               # 'up' | 'down' | 'range'
    pools: Pools
    qm_levels: List[QMLevel]
    atr: float


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().iloc[-1]
    if pd.isna(v):
        v = float((h - l).mean())
    return float(v)


def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[Swing]:
    """Фрактальные свинги с чередованием H/L (из подряд идущих одноимённых
    остаётся более экстремальный)."""
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    raw: List[Swing] = []
    for i in range(left, n - right):
        hw = h[i - left:i + right + 1]
        lw = l[i - left:i + right + 1]
        if h[i] >= hw.max() and int(np.argmax(hw)) == left:
            raw.append(Swing(i, df.index[i], float(h[i]), "H"))
        if l[i] <= lw.min() and int(np.argmin(lw)) == left:
            raw.append(Swing(i, df.index[i], float(l[i]), "L"))
    raw.sort(key=lambda s: (s.idx, s.kind))

    swings: List[Swing] = []
    for s in raw:
        if swings and swings[-1].kind == s.kind:
            better = s.price > swings[-1].price if s.kind == "H" else s.price < swings[-1].price
            if better:
                swings[-1] = s
        else:
            swings.append(s)
    return swings


def label_swings(swings: List[Swing], tol: float = 0.0) -> List[Swing]:
    """Разметка HH/HL/LL/LH (+EQH/EQL при равенстве в пределах tol)."""
    last = {"H": None, "L": None}
    for s in swings:
        p = last[s.kind]
        if p is None:
            s.label = "H" if s.kind == "H" else "L"
        elif abs(s.price - p.price) <= tol:
            s.label = "EQH" if s.kind == "H" else "EQL"
        elif s.kind == "H":
            s.label = "HH" if s.price > p.price else "LH"
        else:
            s.label = "HL" if s.price > p.price else "LL"
        last[s.kind] = s
    return swings


def infer_trend(df: pd.DataFrame, swings: List[Swing], right: int = 2) -> str:
    """Тренд по слому структуры: закрытие выше последнего подтверждённого
    свинг-хая -> 'up', ниже свинг-лоу -> 'down'. Проход последовательный."""
    trend = "range"
    close = df["close"].values
    last_h: Optional[Swing] = None
    last_l: Optional[Swing] = None
    pi = 0
    for i in range(len(df)):
        while pi < len(swings) and swings[pi].idx + right <= i:
            s = swings[pi]
            if s.kind == "H":
                last_h = s
            else:
                last_l = s
            pi += 1
        if last_h is not None and close[i] > last_h.price:
            trend = "up"
        if last_l is not None and close[i] < last_l.price:
            trend = "down"
    return trend


def external_pools(swings: List[Swing], trend: str) -> Pools:
    """Внешняя ликвидность — два последних пула по тренду.
    В аптренде: последний минимум и максимум, сформированный после него.
    В даунтренде зеркально. В рейндже — просто последние минимум и максимум."""
    highs = [s for s in swings if s.kind == "H"]
    lows = [s for s in swings if s.kind == "L"]
    if not highs or not lows:
        return Pools(None, None)
    last_low, last_high = lows[-1], highs[-1]
    if trend == "up":
        hs_after = [x for x in highs if x.idx > last_low.idx]
        return Pools(low=last_low, high=hs_after[-1] if hs_after else last_high)
    if trend == "down":
        ls_after = [x for x in lows if x.idx > last_high.idx]
        return Pools(low=ls_after[-1] if ls_after else last_low, high=last_high)
    return Pools(low=last_low, high=last_high)


def find_qm(df: pd.DataFrame, swings: List[Swing], lookback_swings: int = 14) -> List[QMLevel]:
    """Quasimodo:
      медвежий — H1 -> L1 -> H2 (H2 снял H1), затем закрытие ниже L1  => QML = H1;
      бычий    — L1 -> H1 -> L2 (L2 снял L1), затем закрытие выше H1 => QML = L1.
    Возвращает последние найденные уровни (используются как конфлюэнс к свипу)."""
    out: List[QMLevel] = []
    close = df["close"].values
    seq = swings[-lookback_swings:]
    for k in range(len(seq) - 2):
        a, b, c3 = seq[k], seq[k + 1], seq[k + 2]
        if a.kind == "H" and b.kind == "L" and c3.kind == "H" and c3.price > a.price:
            seg = close[c3.idx + 1:]
            if seg.size and (seg < b.price).any():
                out.append(QMLevel("sell", qml=a.price, head=c3.price, time=c3.time))
        if a.kind == "L" and b.kind == "H" and c3.kind == "L" and c3.price < a.price:
            seg = close[c3.idx + 1:]
            if seg.size and (seg > b.price).any():
                out.append(QMLevel("buy", qml=a.price, head=c3.price, time=c3.time))
    return out[-6:]


def analyze_structure(df: pd.DataFrame, settings) -> Structure:
    a = atr_value(df)
    sw = find_swings(df, settings.swing_left, settings.swing_right)
    label_swings(sw, tol=a * settings.eq_tolerance_atr)
    trend = infer_trend(df, sw, right=settings.swing_right)
    pools = external_pools(sw, trend)
    qm = find_qm(df, sw)
    return Structure(sw, trend, pools, qm, a)
