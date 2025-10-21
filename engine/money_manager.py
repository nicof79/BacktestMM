"""
engine/money_manager.py

MoneyManager encapsule la logique d'allocation, d'ouverture/fermeture de positions (FIFO),
application de commission et slippage, et tenue d'un historique des trades.

API publique principale :
- MoneyManager(initial_capital, max_alloc_pct, commission=0.0, slippage=0.0)
- try_buy(date, price) -> dict | None
- sell_oldest(date, price) -> dict | None
- close_all(date, price) -> list[dict]
- mark_to_market(price) -> float
- snapshot() -> dict
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import math
import copy

class MoneyManager:
    def __init__(self,
                 initial_capital: float,
                 max_alloc_pct: float = 0.2,
                 commission: float = 0.0,
                 slippage: float = 0.0):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not (0 < max_alloc_pct <= 1):
            raise ValueError("max_alloc_pct must be in (0,1]")

        self.initial_capital: float = float(initial_capital)
        self.max_alloc_pct: float = float(max_alloc_pct)
        self.commission: float = float(commission)
        self.slippage: float = float(slippage)

        self.cash: float = float(initial_capital)
        self.open_positions: List[Dict[str, Any]] = []  # FIFO queue of positions
        self.trade_history: List[Dict[str, Any]] = []

    def _apply_commission_and_slippage_buy(self, qty: int, price: float) -> float:
        raw = qty * price
        slip = abs(self.slippage) * price * qty
        fee = abs(self.commission)
        return raw + slip + fee

    def _apply_commission_and_slippage_sell(self, qty: int, price: float) -> float:
        raw = qty * price
        slip = abs(self.slippage) * price * qty
        fee = abs(self.commission)
        return raw - slip - fee

    def try_buy(self, date, price: float) -> Optional[Dict[str, Any]]:
        """
        Attempt to open a position using up to max_alloc_pct of current cash.
        Returns position dict on success or None if cannot buy (insufficient funds or price invalid).
        Position dict fields: buy_date, buy_price, quantity, invested (incl fees).
        """
        if price <= 0 or not math.isfinite(price):
            return None

        max_order_value = self.cash * self.max_alloc_pct
        qty = max(0, math.floor(max_order_value / price))
        if qty < 1:
            # if price fits remaining cash allow a minimum single share buy
            if price <= self.cash:
                qty = 1
            else:
                return None

        invested = self._apply_commission_and_slippage_buy(qty, price)
        # ensure enough cash for invested amount
        if invested > self.cash:
            # try reduce qty
            while qty > 0 and self._apply_commission_and_slippage_buy(qty, price) > self.cash:
                qty -= 1
            if qty < 1:
                return None
            invested = self._apply_commission_and_slippage_buy(qty, price)

        self.cash -= invested
        pos = {
            "buy_date": date,
            "buy_price": float(price),
            "quantity": int(qty),
            "invested": float(invested)
        }
        self.open_positions.append(pos)
        return copy.deepcopy(pos)

    def sell_oldest(self, date, price: float) -> Optional[Dict[str, Any]]:
        """
        Sell the oldest open position (FIFO). Returns trade dict or None if no open positions.
        Trade dict fields: buy_date, sell_date, buy_price, sell_price, quantity, invested, proceeds, profit
        """
        if not self.open_positions:
            return None
        if price <= 0 or not math.isfinite(price):
            return None

        pos = self.open_positions.pop(0)
        qty = int(pos["quantity"])
        invested = float(pos["invested"])
        proceeds = self._apply_commission_and_slippage_sell(qty, price)
        profit = proceeds - invested
        self.cash += proceeds
        trade = {
            "buy_date": pos["buy_date"],
            "sell_date": date,
            "buy_price": float(pos["buy_price"]),
            "sell_price": float(price),
            "quantity": qty,
            "invested": invested,
            "proceeds": float(proceeds),
            "profit": float(profit)
        }
        self.trade_history.append(copy.deepcopy(trade))
        return trade

    def close_all(self, date, price: float) -> List[Dict[str, Any]]:
        """
        Close all open positions at given price (used at end of backtest).
        Returns list of trade dicts.
        """
        trades = []
        # Sell until no open positions
        while self.open_positions:
            t = self.sell_oldest(date, price)
            if t is None:
                break
            trades.append(t)
        return trades

    def mark_to_market(self, price: float) -> float:
        """
        Compute total equity assuming open positions valued at the given price (no commission/slippage).
        Returns total equity = cash + sum(qty * price).
        """
        if price is None or not math.isfinite(price):
            raise ValueError("price must be a finite number")
        open_value = sum(int(p["quantity"]) * price for p in self.open_positions)
        return float(self.cash + open_value)

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a shallow snapshot of current state for logging/debugging.
        """
        return {
            "cash": float(self.cash),
            "initial_capital": float(self.initial_capital),
            "open_positions_count": len(self.open_positions),
            "open_positions": [dict(p) for p in self.open_positions],
            "trade_count": len(self.trade_history)
        }