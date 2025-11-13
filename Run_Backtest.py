from __future__ import annotations

"""
Run_Backtest.py

Convenience runner for the deep value backtest.

Edit the CONFIG dictionary below with your desired parameters, then simply
hit "Run" in your IDE / editor. No command line flags needed.
"""

from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from Deep_Value_Bot import get_sp_universe
from deep_value_strategy import DeepValueStrategy
from paper_broker import Broker, Market, run_backtest


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Backtest window
    "start": "2023-01-01",
    "end": "2024-12-31",

    # Portfolio
    "initial_cash": 1_000.00,
    "max_positions": 20,
    "rebalance_every_n_days": 5,  # 1 = daily, 5 = weekly-ish

    # Data
    "lookback_days": 365,  # how far back to pull price history before `start`

    # Commissions
    "commission_per_share": 0.005,
    "min_commission": 0.50,
}


# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
def get_universe_prices(
    symbols: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download daily price data for `symbols` between `start` and `end`.

    Parameters
    ----------
    symbols : list[str]
        List of tickers (Yahoo format).
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices, index=dates, columns=symbols.
    """
    if not symbols:
        raise ValueError("No symbols provided for price download.")

    print(f"[Run_Backtest] Downloading prices for {len(symbols)} symbols from {start} to {end}...")
    data = yf.download(
        tickers=list(symbols),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        # Single-ticker case (columns: Open, High, Low, Close, Adj Close, Volume)
        price_df = data[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker: data["Adj Close"] is a DataFrame with columns = symbols
        if "Adj Close" in data.columns.get_level_values(0):
            price_df = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            price_df = data["Close"].copy()
        else:
            raise RuntimeError("Unexpected columns from yfinance (no Close/Adj Close).")
    else:
        # Fallback: maybe we only got a single-column "Close"
        if "Adj Close" in data.columns:
            price_df = data[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})
        elif "Close" in data.columns:
            price_df = data[["Close"]].rename(columns={"Close": symbols[0]})
        else:
            raise RuntimeError("Unexpected columns from yfinance (no Close/Adj Close).")

    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    print(f"[Run_Backtest] Downloaded {price_df.shape[0]} rows of prices.")
    return price_df


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
    lookback_days = int(config["lookback_days"])
    commission_per_share = float(config["commission_per_share"])
    min_commission = float(config["min_commission"])

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
    print(f"[Run_Backtest] Commission: {commission_per_share} /share, min {min_commission}")

    # 1) Universe
    print("[Run_Backtest] Building universe from Deep_Value_Bot.get_sp_universe()...")
    universe = get_sp_universe()
    print(f"[Run_Backtest] Universe has {len(universe)} names before trimming.")

    # DEV/EXPERIMENTAL: trim universe so backtests run fast and don't spam Yahoo.
    # Bump this up later (e.g. 200, 500) once you're happy with behavior.
    MAX_UNIVERSE = 5
    if len(universe) > MAX_UNIVERSE:
        print(f"[Run_Backtest] Trimming universe from {len(universe)} to {MAX_UNIVERSE} tickers for this run.")
        universe = universe[:MAX_UNIVERSE]

    # 2) Price history for the whole universe
    prices = get_universe_prices(
        symbols=universe,
        start=lookback_start_dt.strftime("%Y-%m-%d"),
        end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    # 3) Build trading calendar (intersection of requested dates and available prices)
    calendar = [
        d
        for d in prices.index
        if start_dt <= d <= end_dt
    ]
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
    )
    print("[Run_Backtest] Backtest complete.")

    # 6) Analytics
    compute_analytics(broker)


if __name__ == "__main__":
    main(CONFIG)