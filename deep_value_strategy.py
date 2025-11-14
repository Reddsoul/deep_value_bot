from __future__ import annotations

from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from paper_broker import Broker, Market, Strategy
from Deep_Value_Bot import run_screen, GRAHAM_CONFIG


class DeepValueStrategy(Strategy):
    """
    Graham-style long-only deep value strategy.

    Behavior
    --------
    - On start:
        - Receives the full trading calendar.
    - On each bar:
        - Only acts on rebalance dates (every `rebalance_every_n_days`).
        - Calls run_screen(as_of_date=ts, mode="raw") to get full metrics and
          filter flags.
        - Applies explicit exit rules:
            * Sell if:
                - p_tangible_book > MAX_P_TANGIBLE_BOOK_EXIT or
                - ncav_ratio < MIN_NCAV_RATIO_EXIT
              AND the minimum holding period has been satisfied; OR
                - quality_pass == False (immediate exit), OR
                - fundamentals missing and EXIT_ON_MISSING_FUNDAMENTALS = True.
        - After forced sells:
            * Keeps remaining holdings.
            * If there are empty slots (< max_positions), selects new entries
              from names where passes_all == True, sorted by score (cheapest).
            * Rebalances into equal-weight positions over current holdings
              plus new entries.
        - If nothing qualifies, it happily holds cash and existing positions.
    - On end:
        - Optionally liquidates all remaining positions so P&L is realized.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        max_positions: int = 20,
        rebalance_every_n_days: int = 63,  # ~ quarterly by default
        min_holding_days: int = 180,
        liquidate_on_end: bool = True,
        config_overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        self.universe = universe
        self.max_positions = int(max_positions)
        self.rebalance_every_n_days = int(rebalance_every_n_days)
        self.min_holding_days = int(min_holding_days)
        self.liquidate_on_end = bool(liquidate_on_end)

        # Calendar / bookkeeping
        self.calendar: List[pd.Timestamp] = []
        self._last_rebalance_idx: Optional[int] = None

        # Entry dates for minimum holding period logic
        self.entry_dates: Dict[str, pd.Timestamp] = {}

        # Thresholds (per-instance override of global config if desired)
        self.cfg = dict(GRAHAM_CONFIG)
        if config_overrides:
            self.cfg.update(config_overrides)

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
    
    def _is_rebalance_idx(self, bar_idx: int) -> bool:
        """
        Return True if this bar index should trigger a rebalance.

        Rules
        -----
        - First bar (index 0) always rebalances.
        - Otherwise, we require at least `rebalance_every_n_days` trading
          days to have passed since the last rebalance index.
        """
        if bar_idx == 0:
            return True
        if self._last_rebalance_idx is None:
            return True
        return (bar_idx - self._last_rebalance_idx) >= self.rebalance_every_n_days

    def _compute_equity(self, broker: Broker, market: Market) -> float:
        """Compute total equity = cash + sum(position.quantity * price)."""
        equity = float(broker.cash)
        for sym, pos in broker.positions.items():
            price = market.get_price(sym)
            if price is not None and np.isfinite(price):
                equity += pos.quantity * price
        return float(equity)

    def _days_held(self, symbol: str, ts: pd.Timestamp) -> Optional[int]:
        """Return number of calendar days symbol has been held (None if unknown)."""
        entry_ts = self.entry_dates.get(str(symbol))
        if entry_ts is None:
            return None
        return (pd.to_datetime(ts).normalize() - entry_ts.normalize()).days

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------
    def on_start(
        self,
        broker: Broker,
        market: Market,
        calendar: List[pd.Timestamp],
    ) -> None:
        """
        Receive the full trading calendar at the start of the backtest.
        """
        self.calendar = [pd.to_datetime(d) for d in calendar]
        self._last_rebalance_idx = None
        self.entry_dates.clear()
        print(
            f"[DeepValueStrategy] Starting backtest with {len(self.calendar)} bars, "
            f"max_positions={self.max_positions}, "
            f"rebalance_every_n_days={self.rebalance_every_n_days}, "
            f"min_holding_days={self.min_holding_days}"
        )

    def on_bar(
        self,
        broker: Broker,
        market: Market,
        ts: pd.Timestamp,
        slice_df: pd.DataFrame,
    ) -> None:
        """
        Graham-style rebalance logic with explicit entry/exit rules and
        willingness to hold cash.

        NOTE: Only runs on scheduled rebalance indices; on other days this
        is a no-op and we just let the broker mark to market.
        """
        ts = pd.to_datetime(ts)
        bar_idx = self._idx_for_ts(ts)
        if bar_idx is None:
            return

        # --- Quarterly (or N-day) scheduling guard ---
        if not self._is_rebalance_idx(bar_idx):
            return

        # ---------------------------
        # 1) Run deep value screen (raw)
        # ---------------------------
        try:
            screen_df = run_screen(
                as_of_date=ts,
                universe_override=self.universe,
                mode="raw",
            )
        except Exception as e:
            print(f"[DeepValueStrategy] run_screen failed on {ts.date()}: {e}")
            return

        if screen_df is None or screen_df.empty:
            print("[DeepValueStrategy] Screener returned no data; skipping bar.")
            return

        # Identify symbol column
        symbol_col = None
        for c in ["ticker", "symbol", "Symbol", "Ticker"]:
            if c in screen_df.columns:
                symbol_col = c
                break

        if symbol_col is None:
            print("[DeepValueStrategy] Screener output missing symbol/ticker column; skipping bar.")
            return

        # Restrict candidates to symbols that exist in our price universe
        if slice_df is not None and not slice_df.empty:
            traded_universe = set(map(str, slice_df.columns))
            before_ct = len(screen_df)
            screen_df = screen_df[
                screen_df[symbol_col].astype(str).isin(traded_universe)
            ].copy()
            after_ct = len(screen_df)
            print(
                f"[DeepValueStrategy] Universe restricted to traded names: "
                f"{before_ct} -> {after_ct}"
            )
            if screen_df.empty:
                print(
                    f"[DeepValueStrategy] Screener produced {before_ct} names on {ts.date()} "
                    "but none are in the price universe; skipping bar."
                )
                return

        # Use symbol as index for quick lookup
        screen_df[symbol_col] = screen_df[symbol_col].astype(str)
        screen_df.set_index(symbol_col, inplace=True, drop=False)

        # For logging: counts AFTER price-universe restriction
        raw_count = len(screen_df)
        mos_count = int(screen_df.get("mos_pass", False).sum()) if "mos_pass" in screen_df.columns else 0
        qual_count = int(screen_df.get("passes_all", False).sum()) if "passes_all" in screen_df.columns else 0
        print(
            f"[DeepValueStrategy] {ts.date()} | Raw: {raw_count} | "
            f"MoS pass: {mos_count} | MoS+quality pass: {qual_count}"
        )

        # ---------------------------
        # 2) Forced exits (value & quality exit rules)
        # ---------------------------
        current_symbols = set(map(str, broker.positions.keys()))
        forced_sells = set()

        max_ptb_exit = self.cfg.get("MAX_P_TANGIBLE_BOOK_EXIT")
        min_ncav_exit = self.cfg.get("MIN_NCAV_RATIO_EXIT")
        exit_on_missing = self.cfg.get("EXIT_ON_MISSING_FUNDAMENTALS", True)

        for sym in list(current_symbols):
            if sym not in screen_df.index:
                if exit_on_missing:
                    forced_sells.add(sym)
                continue

            row = screen_df.loc[sym]
            ptb = float(row.get("p_tangible_book", np.nan))
            ncav_ratio = float(row.get("ncav_ratio", np.nan))
            quality_pass = bool(row.get("quality_pass", False))

            value_exit = False
            if max_ptb_exit is not None and not np.isnan(ptb):
                if ptb > float(max_ptb_exit):
                    value_exit = True
            if min_ncav_exit is not None and not np.isnan(ncav_ratio):
                if ncav_ratio < float(min_ncav_exit):
                    value_exit = True

            quality_exit = not quality_pass

            days_held = self._days_held(sym, ts)
            min_hold_ok = (
                days_held is None or days_held >= self.min_holding_days
            )

            if quality_exit or (value_exit and min_hold_ok):
                forced_sells.add(sym)

        # Execute forced sells
        for sym in forced_sells:
            qty = broker.get_position(sym)
            if qty > 0:
                print(
                    f"[DeepValueStrategy] {ts.date()} FORCED SELL {sym}: "
                    f"qty={qty}"
                )
                broker.sell(ts, sym, qty)
            self.entry_dates.pop(sym, None)

        # Refresh current holdings after forced sells
        current_symbols = set(map(str, broker.positions.keys()))

        # ---------------------------
        # 3) New entries from Graham-filtered candidates
        # ---------------------------
        if "passes_all" in screen_df.columns:
            candidates = screen_df[screen_df["passes_all"]].copy()
        else:
            candidates = screen_df.copy()

        if "score" in candidates.columns:
            candidates.sort_values("score", inplace=True)

        available_slots = max(0, self.max_positions - len(current_symbols))
        new_buys: List[str] = []
        if available_slots > 0 and not candidates.empty:
            for sym in candidates.index:
                if sym in current_symbols:
                    continue
                new_buys.append(sym)
                if len(new_buys) >= available_slots:
                    break

        if new_buys:
            print(
                f"[DeepValueStrategy] {ts.date()} | New Graham entries: "
                f"{', '.join(new_buys)}"
            )

        target_symbols = list(sorted(current_symbols | set(new_buys)))
        if not target_symbols:
            print(f"[DeepValueStrategy] {ts.date()} | No holdings; staying in cash.")
            self._last_rebalance_idx = bar_idx
            return

        # ---------------------------
        # 4) Rebalance holdings to equal weight
        # ---------------------------
        equity = self._compute_equity(broker, market)
        if equity <= 0:
            print(f"[DeepValueStrategy] {ts.date()} | Non-positive equity, skipping rebalance.")
            self._last_rebalance_idx = bar_idx
            return

        target_pos_count = len(target_symbols)
        target_value_per_pos = equity / float(target_pos_count)

        for sym in target_symbols:
            price = market.get_price(sym)
            if price is None or not np.isfinite(price) or price <= 0:
                continue

            current_qty = broker.get_position(sym)
            current_value = current_qty * price
            delta_value = target_value_per_pos - current_value

            shares_to_trade = int(delta_value // price)
            if shares_to_trade > 0:
                broker.buy(ts, sym, shares_to_trade)
                if current_qty == 0 and shares_to_trade > 0:
                    self.entry_dates[sym] = ts
            elif shares_to_trade < 0:
                broker.sell(ts, sym, -shares_to_trade)

        # Record this bar as the latest rebalance
        self._last_rebalance_idx = bar_idx

    def on_end(self, broker: Broker, market: Market) -> None:
        """
        At the end of the backtest, optionally liquidate everything so we
        get realized P&L.
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
        self.entry_dates.clear()