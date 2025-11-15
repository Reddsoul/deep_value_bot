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

        self.filing_events_by_date: Dict[pd.Timestamp, set[str]] = {}

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
        filing_events_by_date: Optional[Dict[pd.Timestamp, set[str]]] = None,
    ) -> None:
        """
        Receive the full trading calendar and optional SEC filing event map.
        """
        self.calendar = [pd.to_datetime(d) for d in calendar]
        self._last_rebalance_idx = None
        self.entry_dates.clear()
        self.filing_events_by_date = filing_events_by_date or {}

        print(
            f"[DeepValueStrategy] Starting backtest with {len(self.calendar)} bars, "
            f"max_positions={self.max_positions}, "
            f"rebalance_every_n_days={self.rebalance_every_n_days}, "
            f"min_holding_days={self.min_holding_days}"
        )
        if self.filing_events_by_date:
            print(
                f"[DeepValueStrategy] SEC filing event map loaded with "
                f"{len(self.filing_events_by_date)} effective_trade_dates."
            )

    def on_bar(
        self,
        broker: Broker,
        market: Market,
        ts: pd.Timestamp,
        slice_df: pd.DataFrame,
        filing_symbols_today: Optional[set[str]] = None,
    ) -> None:
        """
        Graham-style rebalance logic with:

        - Scheduled rebalances every `rebalance_every_n_days`.
        - Filing-driven decision days triggered by SEC filings (10-K/Q, 8-K, 20-F).
        - Strict min_holding_days on scheduled exits.
        - Filing-driven early exits allowed only for symbols that filed.
        - Cash-aware sizing for better scale-invariance.
        """
        ts = pd.to_datetime(ts)
        bar_idx = self._idx_for_ts(ts)
        if bar_idx is None:
            return

        filing_symbols_today = {str(s).upper() for s in (filing_symbols_today or set())}

        # Optional: log filings that are outside the current strategy universe
        if filing_symbols_today and self.universe is not None:
            universe_set = {str(s).upper() for s in self.universe}
            unknown_syms = filing_symbols_today - universe_set
            if unknown_syms:
                print(
                    f"[DeepValueStrategy] {ts.date()} | Filing events for symbols "
                    f"outside trading universe (ignored): {sorted(unknown_syms)}"
                )

        is_scheduled_rebalance = self._is_rebalance_idx(bar_idx)
        is_filing_decision_day = bool(filing_symbols_today)

        # Decision day rules:
        # - If scheduled: full portfolio rebalance (regardless of filings).
        # - Else if filing-only: local decisions only for filing symbols.
        if not is_scheduled_rebalance and not is_filing_decision_day:
            return

        mode = "scheduled" if is_scheduled_rebalance else "filing"
        if mode == "scheduled":
            print(f"[DeepValueStrategy] Scheduled rebalance at index {bar_idx}, date {ts.date()}.")
        else:
            print(
                f"[DeepValueStrategy] Filing-driven decision day at index {bar_idx}, "
                f"date {ts.date()}, symbols={sorted(filing_symbols_today)}"
            )

        # ------------------------------------------------------------------
        # 1) Run deep value screen (raw, with full metrics & flags)
        # ------------------------------------------------------------------
        try:
            screen_df = run_screen(
                as_of_date=ts,
                universe_override=self.universe,
                mode="raw",
            )
        except Exception as e:
            print(f"[DeepValueStrategy] run_screen failed on {ts.date()}: {e}")
            self._last_rebalance_idx = bar_idx
            return

        if screen_df is None or screen_df.empty:
            print(f"[DeepValueStrategy] Screener returned no data on {ts.date()}; skipping bar.")
            self._last_rebalance_idx = bar_idx
            return

        # Identify symbol column in screener output
        symbol_col = None
        for c in ["ticker", "symbol", "Symbol", "Ticker"]:
            if c in screen_df.columns:
                symbol_col = c
                break
        if symbol_col is None:
            print("[DeepValueStrategy] Screener output missing symbol/ticker column; skipping bar.")
            self._last_rebalance_idx = bar_idx
            return

        # Restrict to traded universe (intersection with price columns)
        screen_df = screen_df.copy()
        screen_df[symbol_col] = screen_df[symbol_col].astype(str)
        before_ct = len(screen_df)

        if slice_df is not None and not slice_df.empty:
            traded_universe = set(map(str, slice_df.columns))
            screen_df = screen_df[
                screen_df[symbol_col].isin(traded_universe)
            ].copy()

        after_ct = len(screen_df)
        print(
            f"[DeepValueStrategy] Universe restricted to traded names: "
            f"{before_ct} -> {after_ct}"
        )
        if screen_df.empty:
            print(
                f"[DeepValueStrategy] Screener produced {before_ct} names on {ts.date()} "
                "but none are in the price universe; staying in cash."
            )
            self._last_rebalance_idx = bar_idx
            return

        screen_df.set_index(symbol_col, inplace=True, drop=False)

        raw_count = len(screen_df)
        mos_count = int(screen_df.get("mos_pass", False).sum()) if "mos_pass" in screen_df.columns else 0
        qual_count = int(screen_df.get("passes_all", False).sum()) if "passes_all" in screen_df.columns else 0
        print(
            f"[DeepValueStrategy] {ts.date()} | Raw: {raw_count} | "
            f"MoS pass: {mos_count} | MoS+quality pass: {qual_count}"
        )

        # ------------------------------------------------------------------
        # 2) Forced exits (value & quality; scheduled vs filing logic)
        # ------------------------------------------------------------------
        current_symbols = {str(sym).upper() for sym in broker.positions.keys()}
        forced_sells: set[str] = set()

        max_ptb_exit = self.cfg.get("MAX_P_TANGIBLE_BOOK_EXIT")
        min_ncav_exit = self.cfg.get("MIN_NCAV_RATIO_EXIT")
        exit_on_missing = self.cfg.get("EXIT_ON_MISSING_FUNDAMENTALS", True)

        for sym_u in list(current_symbols):
            reason_parts: list[str] = []

            # For filing-driven days, we only re-evaluate filing symbols.
            if mode == "filing" and sym_u not in filing_symbols_today:
                continue

            if sym_u not in screen_df.index:
                if exit_on_missing:
                    forced_sells.add(sym_u)
                    reason_parts.append("missing in screen")
                    print(
                        f"[DeepValueStrategy] {ts.date()} FORCED SELL CANDIDATE {sym_u} "
                        f"(reason: missing in screen, mode={mode})"
                    )
                continue

            row = screen_df.loc[sym_u]
            ptb = float(row.get("p_tangible_book", np.nan))
            ncav_ratio = float(row.get("ncav_ratio", np.nan))
            quality_pass = bool(row.get("quality_pass", False))

            value_exit = False
            if max_ptb_exit is not None and not np.isnan(ptb) and ptb > float(max_ptb_exit):
                value_exit = True
                reason_parts.append(f"p_tangible_book {ptb:.2f} > {max_ptb_exit}")
            if min_ncav_exit is not None and not np.isnan(ncav_ratio) and ncav_ratio < float(min_ncav_exit):
                value_exit = True
                reason_parts.append(f"ncav_ratio {ncav_ratio:.2f} < {min_ncav_exit}")

            quality_exit = not quality_pass
            if quality_exit:
                reason_parts.append("quality_pass=False")

            days_held = self._days_held(sym_u, ts)

            if mode == "scheduled":
                # Scheduled: enforce min_holding_days for ALL exits
                min_hold_ok = days_held is not None and days_held >= self.min_holding_days
                should_exit = min_hold_ok and (quality_exit or value_exit)
                exit_kind = "SCHEDULED_EXIT"
            else:
                # Filing-driven: allow early exit ONLY for filing symbols
                # that fail quality / value tests.
                min_hold_ok = True  # overridden by filing logic
                should_exit = quality_exit or value_exit
                exit_kind = "FILING_EVENT_EXIT"

            if should_exit:
                forced_sells.add(sym_u)
                print(
                    f"[DeepValueStrategy] {ts.date()} {exit_kind} {sym_u} "
                    f"(held {days_held} days, reasons: {', '.join(reason_parts)})"
                )

        pre_trades = len(broker.trades)
        for sym_u in forced_sells:
            qty = broker.get_position(sym_u)
            if qty > 0:
                print(
                    f"[DeepValueStrategy] {ts.date()} FORCED SELL {sym_u}: qty={qty}"
                )
                broker.sell(ts, sym_u, qty)
            self.entry_dates.pop(sym_u, None)

        current_symbols = {str(sym).upper() for sym in broker.positions.keys()}

        # ------------------------------------------------------------------
        # 3) New entries
        # ------------------------------------------------------------------
        if "passes_all" in screen_df.columns:
            candidates = screen_df[screen_df["passes_all"]].copy()
        else:
            candidates = screen_df.copy()

        if "score" in candidates.columns:
            candidates.sort_values("score", inplace=True)

        # For filing-driven days, restrict candidate entries to filing symbols only.
        if mode == "filing":
            candidates = candidates[
                candidates.index.astype(str).str.upper().isin(filing_symbols_today)
            ]

        available_slots = max(0, self.max_positions - len(current_symbols))
        new_buys: List[str] = []
        if available_slots > 0 and not candidates.empty:
            for sym_u in candidates.index.astype(str):
                if sym_u.upper() in current_symbols:
                    continue
                new_buys.append(sym_u.upper())
                if len(new_buys) >= available_slots:
                    break

        if new_buys:
            print(
                f"[DeepValueStrategy] {ts.date()} | New Graham entries ({mode}): "
                f"{', '.join(sorted(new_buys))}"
            )

        # Target holdings set for conceptual equal-weighting
        target_symbols = sorted(current_symbols | set(new_buys))
        if not target_symbols:
            print(f"[DeepValueStrategy] {ts.date()} | No holdings; staying in cash on this decision day.")
            if mode == "scheduled":
                self._last_rebalance_idx = bar_idx
            return

        equity = self._compute_equity(broker, market)
        if equity <= 0 or not np.isfinite(equity):
            print(f"[DeepValueStrategy] {ts.date()} | Invalid equity ({equity}), skipping rebalance.")
            if mode == "scheduled":
                self._last_rebalance_idx = bar_idx
            return

        target_pos_count = len(target_symbols)
        target_value_per_pos = equity / float(target_pos_count)

        # On scheduled rebalances, adjust ALL target symbols.
        # On filing-driven days, adjust ONLY filing-related symbols + new entries.
        if mode == "scheduled":
            rebalance_universe = target_symbols
        else:
            rebalance_universe = sorted(
                (current_symbols & filing_symbols_today) | set(new_buys)
            )

        # ------------------------------------------------------------------
        # 4) Cash-aware rebalancing for better scale-invariance
        # ------------------------------------------------------------------
        for sym_u in rebalance_universe:
            price = market.get_price(sym_u)
            if price is None or not np.isfinite(price) or price <= 0:
                print(
                    f"[DeepValueStrategy] {ts.date()} | Skipping {sym_u} in rebalance "
                    f"(invalid price: {price})."
                )
                continue

            current_qty = broker.get_position(sym_u)
            current_value = current_qty * price
            delta_value = target_value_per_pos - current_value

            # Integer shares based on target value
            shares_to_trade = int(delta_value // price)

            # BUY side with cash-aware cap
            if shares_to_trade > 0:
                # Rough safety margin for commission (e.g., extra 1%)
                est_total_cost = shares_to_trade * price * 1.01
                if est_total_cost > broker.cash:
                    max_affordable = int(broker.cash // (price * 1.01))
                    if max_affordable <= 0:
                        print(
                            f"[DeepValueStrategy] {ts.date()} | Cannot afford any shares of {sym_u} "
                            f"(price={price:.4f}, cash={broker.cash:.2f}); skipping BUY."
                        )
                        continue
                    print(
                        f"[DeepValueStrategy] {ts.date()} | Capping BUY for {sym_u} to "
                        f"{max_affordable} shares due to cash constraint."
                    )
                    shares_to_trade = max_affordable

                if shares_to_trade > 0:
                    broker.buy(ts, sym_u, shares_to_trade)
                    if current_qty == 0:
                        self.entry_dates[sym_u] = ts

            # SELL side
            elif shares_to_trade < 0:
                broker.sell(ts, sym_u, -shares_to_trade)

        post_trades = len(broker.trades)
        if post_trades == pre_trades:
            print(
                f"[DeepValueStrategy] {ts.date()} | {mode} decision produced zero trades "
                "(already near target weights / capital constraints)."
            )

        if mode == "scheduled":
            self._last_rebalance_idx = bar_idx

    def on_end(self, broker: Broker, market: Market) -> None:
        """
        End-of-backtest hook.

        Behavior
        --------
        - If liquidate_on_end is True:
            * Sell all remaining positions at the last known prices
              (calendar's last bar).
            * This forces all P&L to be realized and simplifies analytics.
        - If False: do nothing (positions remain marked to market only).
        """
        if not self.liquidate_on_end:
            print("[DeepValueStrategy] on_end: liquidate_on_end=False; leaving positions open.")
            return

        if not self.calendar:
            print("[DeepValueStrategy] on_end: empty calendar; nothing to liquidate.")
            return

        last_ts = self.calendar[-1]
        print(
            f"[DeepValueStrategy] on_end: liquidating all positions at {last_ts.date()} "
            f"because liquidate_on_end=True."
        )

        # Ensure market prices are up-to-date for last_ts
        # (run_backtest already does this before the final mark_to_market,
        # but this is safe and idempotent if called twice).
        # We rely on the backtest loop's final market snapshot, so no
        # manual market.update() here.

        for sym, pos in list(broker.positions.items()):
            qty = int(pos.quantity)
            if qty <= 0:
                continue
            print(f"[DeepValueStrategy] Final liquidation SELL {sym}: qty={qty}")
            broker.sell(last_ts, sym, qty)
            self.entry_dates.pop(sym, None)