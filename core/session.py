"""Торговая сессия и границы торгового дня (таймзона Нью-Йорка, DST учитывается).

Все проверки работают с настоящим UTC-моментом:
  * realtime — системные часы (pd.Timestamp.now(tz="UTC"));
  * бэктест — время бара, приведённое к UTC через bars_to_utc()
    (наивные метки считаются временем сервера брокера со смещением
    settings.bars_utc_offset_hours от UTC).
"""
from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd


def bars_to_utc(ts, settings) -> pd.Timestamp:
    """Метка времени бара -> UTC-момент с учётом смещения сервера брокера."""
    t = pd.Timestamp(ts)
    off = float(getattr(settings, "bars_utc_offset_hours", 0.0) or 0.0)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t - pd.Timedelta(hours=off)


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_session(ts_utc, settings) -> bool:
    """True, если UTC-момент попадает в окно [session_start, session_end)
    по времени session_tz. Окно через полночь поддерживается."""
    if not settings.session_enabled:
        return True
    local = pd.Timestamp(ts_utc).tz_convert(ZoneInfo(settings.session_tz))
    cur = local.hour * 60 + local.minute
    lo = _minutes(settings.session_start)
    hi = _minutes(settings.session_end)
    if lo == hi:
        return True
    if lo < hi:
        return lo <= cur < hi
    return cur >= lo or cur < hi


def local_day(ts_utc, settings) -> date:
    """Дата торгового дня (по session_tz) для данного UTC-момента."""
    return pd.Timestamp(ts_utc).tz_convert(ZoneInfo(settings.session_tz)).date()
