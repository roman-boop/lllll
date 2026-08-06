"""Обёртка над пакетом MetaTrader5: подключение, котировки, ордера.

Импорт пакета защищён, поэтому ядро (structure/patterns/engine) можно использовать
в бэктесте на машине без установленного терминала MT5.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # терминал/пакет не установлен — бэктест по CSV всё равно работает
    mt5 = None

log = logging.getLogger("mt5")


def tf_const(tf: str):
    if mt5 is None:
        raise RuntimeError("Пакет MetaTrader5 не установлен (pip install MetaTrader5, Windows)")
    m = {
        "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4, "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return m[tf]


def _to_df(rates) -> Optional[pd.DataFrame]:
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "tick_volume"]]
    return df.rename(columns={"tick_volume": "volume"})


class MT5Client:
    def __init__(self):
        self.connected = False

    # ---------------- подключение ----------------
    def connect(self, login=None, password=None, server=None, path=None) -> bool:
        if mt5 is None:
            log.error("Пакет MetaTrader5 не установлен: pip install MetaTrader5 (только Windows)")
            return False
        kwargs = {}
        if path:
            kwargs["path"] = path
        if login:
            kwargs.update(login=int(login), password=password, server=server)
        if not mt5.initialize(**kwargs):
            log.error("mt5.initialize() ошибка: %s", mt5.last_error())
            return False
        self.connected = True
        info = mt5.account_info()
        if info:
            log.info("Подключено: счёт %s, эквити %.2f %s (%s)",
                     info.login, info.equity, info.currency, info.server)
        return True

    def shutdown(self):
        if mt5 is not None and self.connected:
            mt5.shutdown()
            self.connected = False

    # ---------------- символы / данные ----------------
    def resolve_symbol(self, candidates: List[str]) -> Optional[str]:
        """Ищет первый доступный у брокера тикер из списка алиасов."""
        for name in candidates:
            si = mt5.symbol_info(name)
            if si is not None:
                if not si.visible:
                    mt5.symbol_select(name, True)
                return name
        for name in candidates:
            found = mt5.symbols_get(f"*{name}*") or []
            for si in found:
                mt5.symbol_select(si.name, True)
                return si.name
        return None

    def rates(self, symbol: str, tf: str, count: int,
              closed_only: bool = True) -> Optional[pd.DataFrame]:
        data = mt5.copy_rates_from_pos(symbol, tf_const(tf), 0, count + 1)
        df = _to_df(data)
        if df is None:
            return None
        return df.iloc[:-1] if closed_only else df  # последний бар — формирующийся

    def rates_range(self, symbol: str, tf: str, dt_from, dt_to) -> Optional[pd.DataFrame]:
        data = mt5.copy_rates_range(symbol, tf_const(tf), dt_from, dt_to)
        return _to_df(data)

    # ---------------- счёт / позиции ----------------
    def equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def positions(self, symbol: Optional[str] = None, magic: Optional[int] = None):
        pos = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        pos = list(pos or [])
        if magic is not None:
            pos = [p for p in pos if p.magic == magic]
        return pos

    def loss_per_lot(self, symbol: str, direction: str, entry: float, sl: float) -> Optional[float]:
        """Денежный убыток на 1 лот при движении entry -> sl (для расчёта объёма)."""
        otype = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        p = mt5.order_calc_profit(otype, symbol, 1.0, float(entry), float(sl))
        if p is not None and p != 0:
            return abs(p)
        si = mt5.symbol_info(symbol)
        if si is None or not si.trade_tick_size:
            return None
        return abs(entry - sl) / si.trade_tick_size * si.trade_tick_value

    # ---------------- ордера ----------------
    def order_market(self, symbol: str, direction: str, lot: float, sl: float, tp: float,
                     magic: int, comment: str = "", deviation: int = 30) -> bool:
        si = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if si is None or tick is None:
            log.error("Нет информации по символу %s", symbol)
            return False
        price = tick.ask if direction == "buy" else tick.bid
        otype = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": otype,
            "price": price,
            "sl": round(float(sl), si.digits),
            "tp": round(float(tp), si.digits),
            "deviation": deviation,
            "magic": magic,
            "comment": comment[:26],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        # перебираем режимы исполнения — у брокеров они разные
        for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            req["type_filling"] = filling
            res = mt5.order_send(req)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info("Ордер исполнен: %s %s %.2f лот @ %.5f (тикет %s)",
                         direction.upper(), symbol, lot, res.price, res.order)
                return True
            if res is not None and res.retcode not in (
                    mt5.TRADE_RETCODE_INVALID_FILL,):
                log.error("order_send отклонён: retcode=%s %s", res.retcode, res.comment)
                return False
        log.error("order_send: не удалось подобрать режим исполнения (filling mode)")
        return False

    def close_position(self, position, comment: str = "daily-loss-stop") -> bool:
        """Закрытие позиции встречным рыночным ордером."""
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return False
        if position.type == mt5.POSITION_TYPE_BUY:
            otype, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            otype, price = mt5.ORDER_TYPE_BUY, tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": otype,
            "price": price,
            "deviation": 50,
            "magic": position.magic,
            "comment": comment[:26],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            req["type_filling"] = filling
            res = mt5.order_send(req)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info("Позиция %s (%s) закрыта: %s", position.ticket,
                         position.symbol, comment)
                return True
        log.error("Не удалось закрыть позицию %s (%s)", position.ticket, position.symbol)
        return False

    def close_all(self, magic: int, comment: str = "daily-loss-stop") -> None:
        for p in self.positions(magic=magic):
            self.close_position(p, comment)

    def modify_sl(self, position, new_sl: float) -> bool:
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": float(new_sl),
            "tp": position.tp,
        }
        res = mt5.order_send(req)
        return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
