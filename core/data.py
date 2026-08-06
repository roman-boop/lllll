"""Загрузка исторических данных из CSV и ресемплинг OHLC.

Поддерживаются:
  * экспорт из MT5 (Вид -> Символы -> Бары -> Экспорт): колонки <DATE> <TIME> <OPEN> ...
  * обычный CSV: time/datetime/date[,time],open,high,low,close[,volume]
Разделитель (табуляция / ; / ,) определяется автоматически.
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ["open", "high", "low", "close"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"])
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"{path}: не найдена колонка времени (date/time/datetime)")

    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"{path}: нет колонки '{c}'")

    out = df[REQUIRED + (["volume"] if "volume" in df.columns else [])].copy()
    out.index = pd.DatetimeIndex(ts)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.astype(float)


def resample_ohlc(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
