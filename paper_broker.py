from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


@dataclass
class Position:
    """
    Single long equity position.
    """
    symbol: str
    quantity: int
    avg_price: float


class Market:
    """
    Simple snapshot of current market prices.

    The backtest loop will update this each bar using a row of the price
    DataFrame (index=symbol, value=price).
    """

    def __init__(self) -> None:
        self.prices: Dict[str, float] = {}

    def update(self, price_series: pd.Series) -> None:
        """
        Update the current prices from a pandas Series.

        Parameters
        ----------
        price_series : pd.Series
            Index = symbols, values = prices (e.g. adjusted close).
        """
        for sym, px in price_series.items():
            if px is None or not np.isfinite(px):
                continue
            self.prices[str(sym)] = float(px)

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Return the last known price for `symbol` (or None if unknown).
        """
        return self.prices.get(str(symbol))


class Broker:
    """
    Minimal paper broker for backtesting.

    - Tracks cash and long-only equity positions.
    - Executes market orders at the current `Market` price.
    - Charges a configurable per-share commission with a minimum per trade.
    - Maintains a history of equity over time and a detailed trade log.
    """

    def __init__(
        self,
        initial_cash: float,
        market: Market,
        commission_per_share: float = 0.005,
        min_commission: float = 0.50,
    ) -> None:
        self.cash = float(initial_cash)
        self.market = market
        self.commission_per_share = float(commission_per_share)
        self.min_commission = float(min_commission)

        self.positions: Dict[str, Position] = {}
        self.equity_history: List[Dict] = []
        self.trades: List[Dict] = []  # dict(timestamp, symbol, side, qty, price, commission, realized_pnl)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _calc_commission(self, quantity: int, price: float) -> float:
        """
        Interactive-Brokers-style: per-share commission with minimum.
        """
        gross = abs(int(quantity)) * self.commission_per_share
        return max(gross, self.min_commission)

    def get_position(self, symbol: str) -> int:
        """
        Current share count for `symbol` (0 if flat).
        """
        pos = self.positions.get(str(symbol))
        return int(pos.quantity) if pos is not None else 0

    def _execute_trade(self, ts: pd.Timestamp, symbol: str, quantity: int) -> None:
        """
        Market order for 'quantity' shares at current market price.

        Parameters
        ----------
        ts : pd.Timestamp
            Timestamp of the bar when the trade is executed.
        symbol : str
            Ticker symbol.
        quantity : int
            Signed quantity: positive = buy, negative = sell.
        """
        if quantity == 0:
            return

        symbol = str(symbol)
        qty = int(quantity)
        side = "BUY" if qty > 0 else "SELL"
        qty = abs(qty)

        price = self.market.get_price(symbol)
        if price is None or not np.isfinite(price) or price <= 0:
            # No valid price -> cannot trade
            return

        commission = self._calc_commission(qty, price)
        realized_pnl = 0.0

        prev_pos = self.positions.get(symbol)

        # ---------------------------
        # BUY
        # ---------------------------
        if side == "BUY":
            cost = qty * price + commission
            if self.cash < cost:
                # insufficient buying power
                return

            if prev_pos is None:
                self.positions[symbol] = Position(symbol=symbol, quantity=qty, avg_price=price)
            else:
                total_shares = prev_pos.quantity + qty
                if total_shares <= 0:
                    # should not happen for buys, but guard anyway
                    return
                new_avg = (prev_pos.avg_price * prev_pos.quantity + price * qty) / total_shares
                prev_pos.quantity = total_shares
                prev_pos.avg_price = new_avg

            self.cash -= cost

        # ---------------------------
        # SELL
        # ---------------------------
        else:
            if prev_pos is None or prev_pos.quantity <= 0 or prev_pos.quantity < qty:
                # no position or trying to go short -> disallow
                return

            # Realized PnL based on position's avg_price
            realized_pnl = (price - prev_pos.avg_price) * qty - commission

            # Update / close position
            remaining = prev_pos.quantity - qty
            if remaining > 0:
                prev_pos.quantity = remaining
            else:
                # Fully closed
                del self.positions[symbol]

            # Receive proceeds
            proceeds = qty * price - commission
            self.cash += proceeds

        # Record trade
        self.trades.append(
            {
                "timestamp": pd.to_datetime(ts),
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "price": float(price),
                "commission": commission,
                "realized_pnl": realized_pnl,
            }
        )

    # ------------------------------------------------------------------
    # Public order interface
    # ------------------------------------------------------------------
    def buy(self, ts: pd.Timestamp, symbol: str, quantity: int) -> None:
        """
        Convenience wrapper for a market BUY.
        """
        self._execute_trade(ts, symbol, quantity)

    def sell(self, ts: pd.Timestamp, symbol: str, quantity: int) -> None:
        """
        Convenience wrapper for a market SELL.
        """
        self._execute_trade(ts, symbol, -abs(int(quantity)))

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------
    def mark_to_market(self, ts: pd.Timestamp) -> None:
        """
        Compute current equity at timestamp `ts` and append to history.
        """
        equity = float(self.cash)
        positions_value = 0.0
        for pos in self.positions.values():
            price = self.market.get_price(pos.symbol)
            if price is not None and np.isfinite(price):
                positions_value += pos.quantity * price

        equity += positions_value
        self.equity_history.append(
            {
                "timestamp": pd.to_datetime(ts),
                "equity": equity,
                "cash": float(self.cash),
                "positions_value": positions_value,
            }
        )


class Strategy(ABC):
    """
    Base class for strategies that plug into the backtest engine.
    """

    @abstractmethod
    def on_start(
        self,
        broker: Broker,
        market: Market,
        calendar: List[pd.Timestamp],
    ) -> None:
        ...

    @abstractmethod
    def on_bar(
        self,
        broker: Broker,
        market: Market,
        ts: pd.Timestamp,
        slice_df: pd.DataFrame,
    ) -> None:
        ...

    @abstractmethod
    def on_end(self, broker: Broker, market: Market) -> None:
        ...


# ----------------------------------------------------------------------
# Backtest loop
# ----------------------------------------------------------------------
def run_backtest(
    prices: pd.DataFrame,
    calendar: List[pd.Timestamp],
    strategy: Strategy,
    broker: Broker,
    market: Market,
) -> Broker:
    """
    Generic backtest loop.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data (e.g. Adjusted Close).
        index = dates, columns = symbols.
    calendar : list[pd.Timestamp]
        Trading calendar for the backtest.
    strategy : Strategy
        Strategy instance.
    broker : Broker
        Broker instance.
    market : Market
        Market instance.

    Returns
    -------
    Broker
        The same broker instance, for convenience.
    """
    # Ensure datetime index
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    calendar = [pd.to_datetime(d) for d in calendar]

    strategy.on_start(broker, market, calendar)

    for ts in calendar:
        if ts not in prices.index:
            # No data for this day (holiday, etc.)
            continue

        # Update market snapshot for this bar
        price_series = prices.loc[ts]
        market.update(price_series)

        # Historical slice up to current ts
        slice_df = prices.loc[:ts]

        # Let the strategy trade
        strategy.on_bar(broker, market, ts, slice_df)

        # Mark-to-market at close
        broker.mark_to_market(ts)

    # Final callback
    strategy.on_end(broker, market)

    return broker