"""Исполнение сигналов: расчёт лота от риска, контроль позиций, безубыток."""
from __future__ import annotations

import logging
import math
from typing import Optional

from config import Settings, MAGIC
from core.engine import Signal
from core.mt5_client import MT5Client, mt5

log = logging.getLogger("trader")


class Trader:
    def __init__(self, client: MT5Client, settings: Settings, dry_run: bool = False):
        self.c = client
        self.s = settings
        self.dry = dry_run

    # ---------------- объём от риска ----------------
    def _lot(self, symbol: str, direction: str, entry: float, sl: float) -> Optional[float]:
        si = mt5.symbol_info(symbol)
        if si is None:
            return None
        lpl = self.c.loss_per_lot(symbol, direction, entry, sl)
        if not lpl:
            return None
        risk_money = self.c.equity() * self.s.risk_per_trade
        if risk_money <= 0:
            return None
        lot = risk_money / lpl
        step = si.volume_step or 0.01
        lot = math.floor(lot / step) * step
        if lot < si.volume_min:
            log.warning("[%s] расчётный лот %.4f меньше минимального %.2f — пропуск",
                        symbol, lot, si.volume_min)
            return None
        return round(min(lot, si.volume_max), 2)

    # ---------------- исполнение сигнала ----------------
    def execute(self, sig: Signal) -> bool:
        if self.s.one_position_per_symbol and self.c.positions(sig.symbol, MAGIC):
            log.info("[%s] позиция системы уже открыта — сигнал пропущен", sig.symbol)
            return False
        lot = self._lot(sig.symbol, sig.direction, sig.entry, sig.sl)
        if lot is None:
            log.error("[%s] не удалось рассчитать лот", sig.symbol)
            return False
        log.info("[%s] ИСПОЛНЕНИЕ %s lot=%.2f sl=%.5f tp=%.5f rr=%.2f (%s)",
                 sig.symbol, sig.direction.upper(), lot, sig.sl, sig.tp, sig.rr, sig.comment)
        if self.dry:
            log.info("DRY-RUN: ордер не отправлен")
            return True
        return self.c.order_market(sig.symbol, sig.direction, lot, sig.sl, sig.tp,
                                   MAGIC, sig.comment, self.s.deviation_points)

    # ---------------- сопровождение: безубыток ----------------
    def manage_breakeven(self):
        r = self.s.breakeven_at_r
        if r <= 0 or self.dry:
            return
        for p in self.c.positions(magic=MAGIC):
            if not p.sl:
                continue
            risk = abs(p.price_open - p.sl)
            if risk <= 0:
                continue
            tick = mt5.symbol_info_tick(p.symbol)
            if tick is None:
                continue
            if p.type == mt5.POSITION_TYPE_BUY:
                if p.sl >= p.price_open:
                    continue  # уже в безубытке
                if tick.bid - p.price_open >= risk * r:
                    if self.c.modify_sl(p, p.price_open):
                        log.info("[%s] SL перенесён в безубыток (+%.1fR)", p.symbol, r)
            else:
                if p.sl <= p.price_open:
                    continue
                if p.price_open - tick.ask >= risk * r:
                    if self.c.modify_sl(p, p.price_open):
                        log.info("[%s] SL перенесён в безубыток (+%.1fR)", p.symbol, r)
