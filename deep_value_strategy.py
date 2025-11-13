from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from paper_broker import Broker, Market, Strategy
from Deep_Value_Bot import run_screen


class DeepValueStrategy(Strategy):
    """
    Simple long-only equal-weight deep value strategy that wraps your screener.

    Behavior
    --------
    - On start:
        - Receives the full trading calendar (list of timestamps).
    - On each bar:
        - Runs the deep value screen for that as-of date.
        - Ranks candidates by `score` (lower = cheaper).
        - Intersects candidates with the symbols that actually exist in the
          backtest price universe (so we only trade names we have data for).
        - Rebalances the portfolio into the top `max_positions` names, with
          equal dollar weights.
    - On end:
        - Optionally liquidates all remaining positions so P&L is realized.
    """

    def __init__(self,
                universe=None,
                max_positions=20,
                rebalance_every_n_days=1,
                liquidate_on_end=True):
        self.universe = universe  # <-- add this
        self.max_positions = int(max_positions)
        self.rebalance_every_n_days = int(rebalance_every_n_days)
        self.liquidate_on_end = bool(liquidate_on_end)
        self.calendar = []
        self._last_rebalance_idx = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _idx_for_ts(self, ts: pd.Timestamp) -> Optional[int]:
        """Return index of `ts` in self.calendar (or None if not found)."""
        if not self.calendar:
            return None
        try:
            return self.calendar.index(pd.to_datetime(ts))
        except ValueError:
            return None

    def _compute_equity(self, broker: Broker, market: Market) -> float:
        """Compute total equity = cash + sum(position.quantity * price)."""
        equity = float(broker.cash)
        for sym, pos in broker.positions.items():
            price = market.get_price(sym)
            if price is not None and np.isfinite(price):
                equity += pos.quantity * price
        return float(equity)

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------
    def on_start(
        self,
        broker: Broker,
        market: Market,
        calendar: List[pd.Timestamp],
    ) -> None:
        self.calendar = [pd.to_datetime(d) for d in calendar]
        self._last_rebalance_idx = None
        print(
            f"[DeepValueStrategy] Starting backtest with {len(self.calendar)} bars, "
            f"max_positions={self.max_positions}, rebalance_every_n_days={self.rebalance_every_n_days}"
        )

    def on_bar(
        self,
        broker: Broker,
        market: Market,
        ts: pd.Timestamp,
        slice_df: pd.DataFrame,
    ) -> None:
        """
        Rebalance into top deep-value names on this bar.

        This calls `run_screen(as_of_date=ts)` to get candidates, then
        intersects them with the symbols that actually have price history
        in the backtest (slice_df.columns). Only those names are eligible
        to trade.
        """
        ts = pd.to_datetime(ts)
        bar_idx = self._idx_for_ts(ts)
        if bar_idx is None:
            return

        # Rebalance only every Nth bar
        if (
            self.rebalance_every_n_days > 1
            and self._last_rebalance_idx is not None
            and bar_idx - self._last_rebalance_idx < self.rebalance_every_n_days
        ):
            return

        # ---------------------------
        # 1) Run deep value screen
        # ---------------------------
        try:
            screen_df = run_screen(as_of_date=ts, universe_override=self.universe)
        except Exception as e:
            print(f"[DeepValueStrategy] run_screen failed on {ts.date()}: {e}")
            return

        if screen_df is None or screen_df.empty:
            return

        # Ensure we have a symbol column
        symbol_col = None
        for c in ["symbol", "ticker", "Symbol", "Ticker"]:
            if c in screen_df.columns:
                symbol_col = c
                break

        if symbol_col is None:
            print("[DeepValueStrategy] Screener output missing symbol/ticker column, skipping bar.")
            return

        # Restrict candidates to symbols that exist in our price universe
        if slice_df is not None and not slice_df.empty:
            traded_universe = set(map(str, slice_df.columns))
            before_ct = len(screen_df)
            screen_df = screen_df[
                screen_df[symbol_col].astype(str).isin(traded_universe)
            ].copy()
            if screen_df.empty:
                print(
                    f"[DeepValueStrategy] Screener produced {before_ct} names on {ts.date()} "
                    "but none are in the price universe used by the backtest; skipping bar."
                )
                return

        # Ensure we have a score column (lower = cheaper)
        if "score" not in screen_df.columns:
            # If your screener doesn't provide a `score`, fall back to P/TB, then NCAV ratio.
            candidates = []
            if "p_tangible_book" in screen_df.columns:
                candidates.append(screen_df["p_tangible_book"])
            if "ncav_ratio" in screen_df.columns:
                candidates.append(screen_df["ncav_ratio"])
            if not candidates:
                print("[DeepValueStrategy] Screener output missing score/p_tangible_book/ncav_ratio, skipping bar.")
                return
            tmp = pd.concat(candidates, axis=1).replace({0.0: np.nan})
            screen_df["score"] = tmp.min(axis=1)

        # Drop rows with missing scores
        sdf = screen_df.dropna(subset=["score"]).copy()
        if sdf.empty:
            return

        sdf.sort_values("score", inplace=True)
        top = sdf.head(self.max_positions)
        target_symbols = list(top[symbol_col].astype(str).unique())
        if not target_symbols:
            return

        target_set = set(target_symbols)
        current_symbols = set(broker.positions.keys())

        # ---------------------------
        # 2) Compute current equity
        # ---------------------------
        equity = self._compute_equity(broker, market)
        if equity <= 0:
            return

        target_pos_count = min(len(target_symbols), self.max_positions)
        if target_pos_count <= 0:
            return

        target_value_per_pos = equity / float(target_pos_count)

        # ---------------------------
        # 3) Exit names that dropped out
        # ---------------------------
        for sym in current_symbols - target_set:
            qty = broker.get_position(sym)
            if qty > 0:
                broker.sell(ts, sym, qty)

        # ---------------------------
        # 4) Rebalance / enter target names
        # ---------------------------
        for sym in target_symbols:
            price = market.get_price(sym)
            if price is None or not np.isfinite(price) or price <= 0:
                continue

            current_qty = broker.get_position(sym)
            current_value = current_qty * price
            delta_value = target_value_per_pos - current_value

            # Convert desired dollar change to shares
            shares_to_trade = int(delta_value // price)
            if shares_to_trade > 0:
                broker.buy(ts, sym, shares_to_trade)
            elif shares_to_trade < 0:
                broker.sell(ts, sym, -shares_to_trade)

        self._last_rebalance_idx = bar_idx

    def on_end(self, broker: Broker, market: Market) -> None:
        """
        At the end of the backtest, optionally liquidate everything so we get realized P&L.
        """
        if not self.liquidate_on_end:
            return

        if not self.calendar:
            ts = pd.Timestamp.utcnow()
        else:
            ts = self.calendar[-1]

        print("[DeepValueStrategy] End of backtest, liquidating remaining positions.")
        for sym, pos in list(broker.positions.items()):
            qty = pos.quantity
            if qty > 0:
                broker.sell(ts, sym, qty)