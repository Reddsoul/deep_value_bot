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
        qty_signed = int(quantity)
        side = "BUY" if qty_signed > 0 else "SELL"
        qty = abs(qty_signed)

        price = self.market.get_price(symbol)
        if price is None or not np.isfinite(price) or price <= 0:
            print(
                f"[Broker] {ts} | Cannot {side} {symbol} (invalid price: {price}). "
                "Trade skipped."
            )
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
                print(
                    f"[Broker] {ts} | Insufficient cash to BUY {qty} {symbol} "
                    f"at {price:.4f} (cost {cost:.2f}, cash {self.cash:.2f}). Trade skipped."
                )
                return

            if prev_pos is None:
                self.positions[symbol] = Position(symbol=symbol, quantity=qty, avg_price=price)
            else:
                total_shares = prev_pos.quantity + qty
                if total_shares <= 0:
                    print(
                        f"[Broker] {ts} | Inconsistent position state on BUY for {symbol}; "
                        f"current qty {prev_pos.quantity}, attempting BUY {qty}. Trade skipped."
                    )
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
                print(
                    f"[Broker] {ts} | Cannot SELL {qty} {symbol}; "
                    f"position qty={0 if prev_pos is None else prev_pos.quantity}. Trade skipped."
                )
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

        Invariant:
            equity ≈ cash + Σ(position.quantity * price)
        """
        positions_value = 0.0
        for pos in self.positions.values():
            price = self.market.get_price(pos.symbol)
            if price is not None and np.isfinite(price):
                positions_value += pos.quantity * price
            else:
                # If we can't price a position, we treat it as zero-value and log.
                print(
                    f"[Broker] {ts} | Missing/invalid price for {pos.symbol} in mark_to_market; "
                    "treating as zero for now."
                )

        equity = float(self.cash) + positions_value

        if not np.isfinite(equity) or equity < -1e-6:
            print(
                f"[Broker] {ts} | WARNING: equity invariant violated or invalid "
                f"(equity={equity}, cash={self.cash}, positions_value={positions_value})."
            )

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
        filing_events_by_date: Optional[Dict[pd.Timestamp, set[str]]] = None,
    ) -> None:
        """
        Called once before the backtest loop starts.

        filing_events_by_date:
            Optional mapping from date -> set of symbols that have
            a filing-effective decision on that trading day.
        """
        ...

    @abstractmethod
    def on_bar(
        self,
        broker: Broker,
        market: Market,
        ts: pd.Timestamp,
        slice_df: pd.DataFrame,
        filing_symbols_today: Optional[set[str]] = None,
    ) -> None:
        """
        Called once per bar in the trading calendar.

        filing_symbols_today:
            Optional set of symbols that have a filing-driven decision
            on this bar (effective_trade_date).
        """
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
    filing_events_by_date: Optional[Dict[pd.Timestamp, set[str]]] = None,
) -> Broker:
    """
    Generic backtest loop with optional SEC filing event support.

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
    filing_events_by_date : dict[pd.Timestamp, set[str]] | None
        Optional mapping from effective_trade_date (normalized date) to
        the set of symbols with filing-driven decision events on that day.

    Returns
    -------
    Broker
        The same broker instance, for convenience.
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    calendar = [pd.to_datetime(d) for d in calendar]

    # Normalize keys of filing_events_by_date to date-only for robust lookup
    norm_events = None
    if filing_events_by_date:
        norm_events = {}
        for dt, syms in filing_events_by_date.items():
            key = pd.to_datetime(dt).normalize()
            norm_events.setdefault(key, set()).update({str(s).upper() for s in syms})

    strategy.on_start(broker, market, calendar, norm_events)

    for ts in calendar:
        if ts not in prices.index:
            continue

        price_series = prices.loc[ts]
        market.update(price_series)

        slice_df = prices.loc[:ts]

        filing_syms_today: Optional[set[str]] = None
        if norm_events is not None:
            filing_syms_today = norm_events.get(ts.normalize(), set())
            if filing_syms_today:
                print(
                    f"[run_backtest] {ts.date()} | Filing-driven decision day for "
                    f"{len(filing_syms_today)} symbols."
                )

        strategy.on_bar(broker, market, ts, slice_df, filing_syms_today)

        broker.mark_to_market(ts)

    strategy.on_end(broker, market)
    return broker