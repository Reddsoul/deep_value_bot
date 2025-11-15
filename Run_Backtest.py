from __future__ import annotations

"""
Run_Backtest.py

Convenience runner for the deep value backtest.

Edit the CONFIG dictionary below with your desired parameters, then simply
hit "Run" in your IDE / editor. No command line flags needed.

This version is tuned for a more Graham-style, low-turnover portfolio:
- Rebalance every ~quarter by default.
- Minimum holding period is explicit.
"""

from typing import List, Dict, Iterable, Tuple, Optional  # extended

import numpy as np
import pandas as pd
import yfinance as yf

from Deep_Value_Bot import get_nasdaq_universe
from deep_value_strategy import DeepValueStrategy
from paper_broker import Broker, Market, run_backtest

from datetime import date
import time  # NEW


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Backtest window
    "start": "2025-01-01",
    "end": date.today().strftime("%Y-%m-%d"),

    # Portfolio
    "initial_cash": 100000.00,
    "max_positions": 20,

    # Rebalancing / holding period
    # rebalance_every_n_days ~ 63 ≈ quarterly (252 trading days / 4)
    "rebalance_every_n_days": 63,
    "min_holding_days": 180,  # enforce minimum holding period

    # Data
    "lookback_days": 365,  # how far back to pull price history before `start`

    # Commissions
    "commission_per_share": 0.005,
    "min_commission": 0.50,

    # Optional SEC filings CSV (symbol, filing_type, filing_date, effective_trade_date)
    "sec_filings_csv": None,  # e.g. "sec_filings.csv"
}

# DEV/EXPERIMENTAL: limit universe size for speed during development
MAX_UNIVERSE = 50000

# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
# In-process price cache: (symbols tuple, start, end) -> DataFrame
# Symbols that repeatedly fail to download prices get quarantined here.
_QUARANTINED_SYMBOLS: set[str] = set()


def filter_universe_by_price_history(
    universe: List[str],
    prices: pd.DataFrame,
    min_history_days: int = 180,
) -> List[str]:
    """
    Filter a symbol universe to those with at least `min_history_days`
    of non-NaN prices in the provided price DataFrame.
    """
    if prices.empty:
        print("[Run_Backtest] WARNING: prices DataFrame is empty in filter_universe_by_price_history.")
        return []

    universe = [str(s).strip().upper() for s in universe if s]
    filtered: List[str] = []

    for sym in universe:
        if sym not in prices.columns:
            continue
        s = prices[sym].dropna()
        if s.empty:
            continue
        span_days = (s.index.max() - s.index.min()).days
        if span_days >= min_history_days:
            filtered.append(sym)

    print(
        f"[Run_Backtest] Price-history filter: {len(universe)} -> {len(filtered)} "
        f"symbols with >= {min_history_days} days of data."
    )
    return filtered

def build_filing_events_by_date(filings_df: pd.DataFrame) -> Dict[pd.Timestamp, set[str]]:
    """
    Build filing_events_by_date from a DataFrame with columns:
        symbol, filing_type, filing_date, effective_trade_date

    Returns
    -------
    dict[pd.Timestamp, set[str]]
        Mapping from effective_trade_date (normalized) to the set of symbols
        with filing-driven decision events that day.
    """
    required_cols = {"symbol", "filing_type", "filing_date", "effective_trade_date"}
    missing = required_cols - set(filings_df.columns)
    if missing:
        raise ValueError(f"filings_df missing required columns: {missing}")

    df = filings_df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["effective_trade_date"] = pd.to_datetime(df["effective_trade_date"])

    events: Dict[pd.Timestamp, set[str]] = {}
    for eff_date, group in df.groupby(df["effective_trade_date"].dt.normalize()):
        syms = set(group["symbol"].tolist())
        events[pd.to_datetime(eff_date)] = syms

    print(
        f"[Run_Backtest] SEC filing events: {len(events)} effective_trade_dates, "
        f"{len(df)} total filing rows."
    )
    return events

_PRICE_CACHE: Dict[Tuple[Tuple[str, ...], str, str], pd.DataFrame] = {}

def _chunked(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def get_universe_prices(
    symbols: List[str],
    start: str,
    end: str,
    batch_size: int = 32,
    max_retries: int = 5,
    base_sleep: float = 2.0,
) -> pd.DataFrame:
    """
    Download daily price data for `symbols` between `start` and `end`
    with batching, retry logic, and an in-process cache plus per-symbol
    quarantine.

    More defensive against yfinance rate limiting and symbol-specific failures.
    """
    if not symbols:
        raise ValueError("No symbols provided for price download.")

    uniq = sorted({str(s).strip().upper() for s in symbols if s})
    cache_key = (tuple(uniq), str(start), str(end))
    if cache_key in _PRICE_CACHE:
        print("[Run_Backtest] Using cached price DataFrame.")
        return _PRICE_CACHE[cache_key].copy()

    print(
        f"[Run_Backtest] Downloading prices for {len(uniq)} symbols "
        f"from {start} to {end} in batches of {batch_size}..."
    )

    all_frames: List[pd.DataFrame] = []
    quarantined_local: set[str] = set()

    def _download_batch(batch_syms: List[str]) -> pd.DataFrame:
        """
        Helper that tries a batch, then halves, then single-symbol fallback.
        Returns a DataFrame (possibly empty) of Adj Close / Close.
        """
        nonlocal quarantined_local

        # First try full batch with retries
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    tickers=batch_syms,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=True,
                )
                if df.empty:
                    raise RuntimeError("Empty DataFrame returned from yfinance.")

                # Single-symbol case
                if len(batch_syms) == 1 and not isinstance(df.columns, pd.MultiIndex):
                    sym = batch_syms[0]
                    if "Adj Close" in df.columns:
                        adj = df["Adj Close"].rename(sym).to_frame()
                    elif "Close" in df.columns:
                        adj = df["Close"].rename(sym).to_frame()
                    else:
                        raise RuntimeError("Unexpected columns for single-symbol download")
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        lvl0 = df.columns.get_level_values(0)
                        if "Adj Close" in lvl0:
                            adj = df["Adj Close"]
                        elif "Close" in lvl0:
                            adj = df["Close"]
                        else:
                            raise RuntimeError("Unexpected MultiIndex columns from yfinance.")
                    else:
                        adj = df

                return adj

            except Exception as e:
                last_err = e
                msg = str(e)
                if "Rate limited" in msg or "Too Many Requests" in msg:
                    sleep_s = base_sleep * (2 ** attempt) * 3.0
                else:
                    sleep_s = base_sleep * (2 ** attempt)

                print(
                    f"[Run_Backtest] Price download failed for batch "
                    f"{batch_syms[0]}..{batch_syms[-1]} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Sleeping {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)

        print(
            f"[Run_Backtest] WARNING: giving up on batch "
            f"{batch_syms[0]}..{batch_syms[-1]} after {max_retries} attempts. "
            f"Last error: {last_err}"
        )

        # If batch failed entirely and has more than 1 symbol, try smaller chunks
        if len(batch_syms) > 1:
            mid = len(batch_syms) // 2
            left = _download_batch(batch_syms[:mid])
            right = _download_batch(batch_syms[mid:])
            # Combine partial results
            if not left.empty or not right.empty:
                return pd.concat([left, right], axis=1)
        else:
            # Single-symbol repeated failure: quarantine it
            sym = batch_syms[0]
            print(f"[Run_Backtest] Quarantining symbol with persistent price failures: {sym}")
            quarantined_local.add(sym)

        return pd.DataFrame()

    # Main loop over top-level batches
    for batch in _chunked(uniq, batch_size=batch_size):
        batch = [s for s in batch if s not in _QUARANTINED_SYMBOLS]
        if not batch:
            continue

        adj = _download_batch(batch)
        if not adj.empty:
            all_frames.append(adj)

        # Gentle pause between batches
        time.sleep(1.0)

    if quarantined_local:
        _QUARANTINED_SYMBOLS.update(quarantined_local)
        print(f"[Run_Backtest] Total newly quarantined symbols: {len(quarantined_local)}")

    if not all_frames:
        raise RuntimeError("[Run_Backtest] No price data downloaded at all (all batches failed).")

    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    cols = [s for s in uniq if s in prices.columns]
    prices = prices[cols]

    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    print(
        f"[Run_Backtest] Downloaded {prices.shape[0]} rows of prices for "
        f"{len(cols)} symbols (after dropping failures & duplicates)."
    )
    _PRICE_CACHE[cache_key] = prices.copy()
    return prices

# ----------------------------------------------------------------------
# Analytics
# ----------------------------------------------------------------------
def compute_analytics(broker: Broker) -> None:
    """
    Compute and print basic performance analytics from broker state.
    """
    eq_df = pd.DataFrame(broker.equity_history)
    if eq_df.empty:
        print("[Analytics] No equity history to analyze.")
        return

    eq_df.sort_values("timestamp", inplace=True)
    eq_df.set_index("timestamp", inplace=True)

    start_equity = float(eq_df["equity"].iloc[0])
    end_equity = float(eq_df["equity"].iloc[-1])
    net_pnl = end_equity - start_equity
    total_return = (end_equity / start_equity) - 1.0 if start_equity > 0 else np.nan

    # How many calendar days in the backtest
    n_days = (eq_df.index[-1] - eq_df.index[0]).days or 1
    annualized_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0 if n_days > 0 else np.nan

    # Max drawdown
    roll_max = eq_df["equity"].cummax()
    drawdown = eq_df["equity"] / roll_max - 1.0
    max_drawdown = drawdown.min()

    # Daily Sharpe ratio (0% risk-free)
    daily_returns = eq_df["equity"].pct_change().dropna()
    if not daily_returns.empty and daily_returns.std(ddof=0) > 0:
        daily_sharpe = np.sqrt(252.0) * daily_returns.mean() / daily_returns.std(ddof=0)
    else:
        daily_sharpe = np.nan

    trades_df = pd.DataFrame(broker.trades)
    if trades_df.empty:
        num_trades = 0
        win_rate = np.nan
        avg_win = np.nan
        avg_loss = np.nan
        avg_pnl_per_trade = np.nan
        profit_factor = np.nan
        top_contrib = pd.Series(dtype=float)
    else:
        closed_trades = trades_df[(trades_df["side"] == "SELL") & trades_df["realized_pnl"].notna()]
        num_trades = int(len(closed_trades))
        wins = closed_trades[closed_trades["realized_pnl"] > 0]
        losses = closed_trades[closed_trades["realized_pnl"] < 0]

        win_rate = len(wins) / num_trades if num_trades > 0 else np.nan
        avg_win = wins["realized_pnl"].mean() if not wins.empty else np.nan
        avg_loss = losses["realized_pnl"].mean() if not losses.empty else np.nan
        avg_pnl_per_trade = closed_trades["realized_pnl"].mean() if num_trades > 0 else np.nan

        gross_profit = wins["realized_pnl"].sum() if not wins.empty else 0.0
        gross_loss = losses["realized_pnl"].sum() if not losses.empty else 0.0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else np.nan

        top_contrib = closed_trades.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)

    print("\n==================== Backtest Summary ====================")
    print(f"Start equity:       {start_equity:,.2f}")
    print(f"End equity:         {end_equity:,.2f}")
    print(f"Net P&L:            {net_pnl:,.2f}")
    print(f"Total return:       {total_return*100:6.2f}%")
    print(f"Annualized return:  {annualized_return*100:6.2f}%")
    print(f"Max drawdown:       {max_drawdown*100:6.2f}%")
    print(f"Sharpe (daily):     {daily_sharpe:6.2f}" if not np.isnan(daily_sharpe) else "Sharpe (daily):     N/A")
    print(f"Number of trades:   {num_trades}")
    print(f"Win rate:           {win_rate*100:6.2f}%" if not np.isnan(win_rate) else "Win rate:           N/A")
    print(f"Average winner:     {avg_win:,.2f}" if not np.isnan(avg_win) else "Average winner:     N/A")
    print(f"Average loser:      {avg_loss:,.2f}" if not np.isnan(avg_loss) else "Average loser:      N/A")
    print(f"Avg P&L / trade:    {avg_pnl_per_trade:,.2f}" if not np.isnan(avg_pnl_per_trade) else "Avg P&L / trade:    N/A")
    print(f"Profit factor:      {profit_factor:6.2f}" if not np.isnan(profit_factor) else "Profit factor:      N/A")

    if 'top_contrib' in locals() and not top_contrib.empty:
        print("\nTop contributors (realized PnL by symbol):")
        print(top_contrib.head(10).to_string(float_format=lambda x: f"{x:,.2f}"))

        print("\nWorst detractors (realized PnL by symbol):")
        print(top_contrib.tail(10).sort_values().to_string(float_format=lambda x: f"{x:,.2f}"))

    print("\nEquity curve (tail):")
    print(eq_df.tail().to_string(float_format=lambda x: f"{x:,.2f}"))
    print("==========================================================\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(config: Dict[str, object]) -> None:
    # Unpack config
    start_str = str(config["start"])
    end_str = str(config["end"])
    initial_cash = float(config["initial_cash"])
    max_positions = int(config["max_positions"])
    rebalance_every_n_days = int(config["rebalance_every_n_days"])
    min_holding_days = int(config["min_holding_days"])
    lookback_days = int(config["lookback_days"])
    commission_per_share = float(config["commission_per_share"])
    min_commission = float(config["min_commission"])

    # Optional: SEC filing events
    sec_filings_csv = config.get("sec_filings_csv")
    filing_events_by_date = None
    if sec_filings_csv:
        print(f"[Run_Backtest] Loading SEC filings from {sec_filings_csv!r}...")
        filings_df = pd.read_csv(sec_filings_csv)
        filing_events_by_date = build_filing_events_by_date(filings_df)
    else:
        print("[Run_Backtest] No sec_filings_csv specified; running without filing events.")

    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    if end_dt < start_dt:
        raise ValueError("CONFIG['end'] date must be >= CONFIG['start'] date")

    lookback_start_dt = start_dt - pd.Timedelta(days=lookback_days)

    print(f"[Run_Backtest] Backtest period: {start_dt.date()} -> {end_dt.date()}")
    print(f"[Run_Backtest] Lookback: {lookback_days} days (from {lookback_start_dt.date()})")
    print(f"[Run_Backtest] Initial cash: {initial_cash:,.2f}")
    print(f"[Run_Backtest] Max positions: {max_positions}")
    print(f"[Run_Backtest] Rebalance every N days: {rebalance_every_n_days}")
    print(f"[Run_Backtest] Min holding days: {min_holding_days}")
    print(f"[Run_Backtest] Commission: {commission_per_share} /share, min {min_commission}")

    # 1) Universe (NASDAQ)
    print("[Run_Backtest] Building universe from Deep_Value_Bot.get_nasdaq_universe()...")
    universe = get_nasdaq_universe()
    print(f"[Run_Backtest] NASDAQ universe has {len(universe)} names before trimming.")

    if len(universe) > MAX_UNIVERSE:
        print(
            f"[Run_Backtest] Trimming universe from {len(universe)} to "
            f"{MAX_UNIVERSE} tickers for this run (dev safety cap)."
        )
        universe = universe[:MAX_UNIVERSE]

    # 2) Price history for the whole universe
    prices = get_universe_prices(
        symbols=universe,
        start=lookback_start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    # 2b) Price-based universe cleaning (must occur before fundamentals)
    universe_clean = filter_universe_by_price_history(
        universe=universe,
        prices=prices,
        min_history_days=180,
    )
    if not universe_clean:
        raise RuntimeError("[Run_Backtest] No symbols with sufficient price history; aborting.")

    universe = universe_clean
    print(f"[Run_Backtest] Final trading universe size after price cleaning: {len(universe)}")

    # 3) Build trading calendar (intersection of requested dates and available prices)
    calendar = [d for d in prices.index if start_dt <= d <= end_dt]
    print(f"[Run_Backtest] Trading calendar has {len(calendar)} bars in requested period.")

    # 4) Instantiate Market, Broker, Strategy
    market = Market()
    broker = Broker(
        initial_cash=initial_cash,
        market=market,
        commission_per_share=commission_per_share,
        min_commission=min_commission,
    )
    strategy = DeepValueStrategy(
        universe=universe,
        max_positions=max_positions,
        rebalance_every_n_days=rebalance_every_n_days,
        min_holding_days=min_holding_days,
        liquidate_on_end=True,
    )

    # 5) Run backtest
    print("[Run_Backtest] Starting backtest...")
    run_backtest(
        prices=prices,
        calendar=calendar,
        strategy=strategy,
        broker=broker,
        market=market,
        filing_events_by_date=filing_events_by_date,
    )
    print("[Run_Backtest] Backtest complete.")

    # 6) Analytics
    compute_analytics(broker)


if __name__ == "__main__":
    main(CONFIG)