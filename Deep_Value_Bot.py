#!/usr/bin/env python3
"""
Deep_Value_Bot.py

Deep value screener that builds its universe from the S&P 400 + S&P 600
Wikipedia pages and pulls fundamentals from Yahoo Finance (via yfinance).

Public API
----------
run_screen(as_of_date: pd.Timestamp | str) -> pd.DataFrame
    Run the deep value screen as of (approximately) `as_of_date` and return
    a DataFrame of candidates with valuation metrics and a `score` column
    where lower = cheaper.

get_sp_universe() -> List[str]
    Return the combined S&P 400 + S&P 600 ticker universe (as Yahoo tickers).

Notes
-----
- This module is intentionally "chatty" in its logging so you can see what it is
  doing (downloading Wikipedia pages, hitting Yahoo, etc.).
- It is *not* optimized for speed. The goal is to keep it readable.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO  # for pd.read_html on literal html strings


_SP_UNIVERSE_CACHE: Optional[List[str]] = None

# Small fallback universe is kept here but NOT used for the mid/small universe.
FALLBACK_UNIVERSE: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def _fetch_sp_table(url: str) -> pd.DataFrame:
    """
    Helper to fetch the FIRST Wikipedia table on `url` that has a 'Symbol' column.

    Wikipedia pages like S&P 600 can contain multiple tables. We don't want
    the first random table; we want the one that actually contains ticker
    symbols. This scans all tables on the page and returns the first one
    where a column named 'Symbol' is present.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DeepValueBot/1.0; +https://example.com/bot)"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    # Future-proof: read_html on a StringIO wrapper to avoid FutureWarning
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError(f"No HTML tables found at {url}")

    for i, tbl in enumerate(tables):
        cols = [str(c).strip().lower() for c in tbl.columns]
        if any(c == "symbol" for c in cols):
            # Found the table we care about
            return tbl

    # If we get here, none of the tables have a 'Symbol' column
    raise RuntimeError(f"No table with 'Symbol' column found at {url}")


def get_sp_universe(force_refresh: bool = False) -> List[str]:
    """
    Fetch S&P 400 + S&P 600 tickers from Wikipedia.

    Returns
    -------
    List[str]
        Sorted list of unique ticker symbols in Yahoo Finance format.

    Behavior
    --------
    - Uses `_fetch_sp_table()` which scans ALL tables on the page and picks
      the one with a 'Symbol' column.
    - Does NOT fall back to FAANG/mega-cap names. If something goes wrong,
      this function raises, because this strategy is specifically intended
      to trade mid/small-cap deep value names.
    """
    global _SP_UNIVERSE_CACHE
    if _SP_UNIVERSE_CACHE is not None and not force_refresh:
        return _SP_UNIVERSE_CACHE

    urls = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600"),
    ]

    tickers: List[str] = []

    for url, label in urls:
        try:
            print(f"[Deep_Value_Bot] Fetching {label} from {url} ...")
            df = _fetch_sp_table(url)
            if "Symbol" not in df.columns:
                raise RuntimeError(f"Table for {label} missing 'Symbol' column.")
            symbols = df["Symbol"].astype(str).str.strip().tolist()
            tickers.extend(symbols)
        except Exception as e:
            raise RuntimeError(
                f"[Deep_Value_Bot] ERROR: Failed to load {label} constituents "
                f"from Wikipedia.\nURL: {url}\nReason: {e}\n\n"
                "Deep value strategy requires mid/small caps — no fallback will be used.\n"
                "Fix your internet or verify Wikipedia is reachable, then retry."
            )

    # Deduplicate + sort
    tickers = sorted(set(tickers))

    if not tickers:
        raise RuntimeError(
            "[Deep_Value_Bot] ERROR: Loaded 0 tickers from S&P 400/600.\n"
            "You must fix Wikipedia access — deep value bot does NOT use "
            "large-cap fallback tickers."
        )

    print(f"[Deep_Value_Bot] Loaded {len(tickers)} tickers from S&P 400 + 600.")
    _SP_UNIVERSE_CACHE = tickers
    return tickers


def _download_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Download fundamental data for a list of tickers via yfinance.

    Returns a DataFrame indexed by ticker, with columns like:
    - market_cap
    - total_assets
    - total_liab
    - net_tangible_assets
    - cash
    - total_debt
    - current_assets
    - current_liabilities
    """
    records: List[Dict] = []
    for i, t in enumerate(tickers, start=1):
        try:
            print(f"[Deep_Value_Bot] [{i}/{len(tickers)}] Fetching fundamentals for {t}...")
            yf_t = yf.Ticker(t)
            info = yf_t.info or {}
        except Exception as e:
            print(f"[Deep_Value_Bot] Error fetching {t}: {e}")
            continue

        rec = {"ticker": t}
        # Extract a few key fields; missing values become NaN
        for key in [
            "marketCap",
            "totalAssets",
            "totalLiab",
            "netTangibleAssets",
            "cash",
            "totalDebt",
            "totalCurrentAssets",
            "totalCurrentLiabilities",
        ]:
            rec[key] = info.get(key, np.nan)

        records.append(rec)

    if not records:
        raise RuntimeError("No fundamental data could be fetched.")

    df = pd.DataFrame.from_records(records).set_index("ticker")
    return df


def _compute_value_metrics(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute various deep value metrics and a combined score.
    """
    df = fund_df.copy()

    # Basic aliases for readability
    mkt = df["marketCap"].astype(float)
    total_assets = df["totalAssets"].astype(float)
    total_liab = df["totalLiab"].astype(float)
    net_tang = df["netTangibleAssets"].astype(float)
    cash = df["cash"].astype(float)
    total_debt = df["totalDebt"].astype(float)
    ca = df["totalCurrentAssets"].astype(float)
    cl = df["totalCurrentLiabilities"].astype(float)

    # Avoid divide-by-zero with np.where
    # 1) Price-to-book (approx): market_cap / net_tangible_assets
    df["p_tangible_book"] = np.where(net_tang > 0, mkt / net_tang, np.nan)

    # 2) NCAV (Net Current Asset Value) = current assets - total liabilities
    ncav = ca - total_liab
    df["ncav"] = ncav
    df["ncav_ratio"] = np.where(mkt > 0, ncav / mkt, np.nan)

    # 3) Simple leverage proxy: total_debt / market_cap
    df["debt_to_equity_proxy"] = np.where(mkt > 0, total_debt / mkt, np.nan)

    # Now build a single "score" where lower = cheaper.
    score_components = []

    # Price-to-tangible-book (cap at some upper bound so outliers don't dominate)
    if "p_tangible_book" in df.columns:
        ptb = df["p_tangible_book"].clip(lower=0.0, upper=10.0)
        score_components.append(ptb)

    # NCAV ratio (we want lower market cap relative to NCAV => more negative is better).
    if "ncav_ratio" in df.columns:
        ncav_r = df["ncav_ratio"]
        # Flip sign so that "cheaper" -> lower positive number
        ncav_score = (-ncav_r).clip(lower=-10.0, upper=10.0)
        score_components.append(ncav_score)

    # Leverage proxy
    if "debt_to_equity_proxy" in df.columns:
        lev = df["debt_to_equity_proxy"].clip(lower=0.0, upper=10.0)
        score_components.append(lev)

    if not score_components:
        raise RuntimeError("No score components were computed; check fundamentals.")

    score_mat = pd.concat(score_components, axis=1)
    df["score"] = score_mat.mean(axis=1)

    return df


def run_screen(as_of_date, universe_override=None):
    if not isinstance(as_of_date, pd.Timestamp):
        as_of_date = pd.to_datetime(as_of_date)

    print(f"[Deep_Value_Bot] Running deep value screen as of {as_of_date.date()}...")

    # ---- KEY CHANGE ----
    if universe_override is not None:
        universe = universe_override
    else:
        universe = get_sp_universe()
    # ---------------------

    if not universe:
        raise RuntimeError("Universe is empty; cannot run screen.")

    fund_df = _download_fundamentals(universe)
    val_df = _compute_value_metrics(fund_df)

    val_df = val_df.sort_values("score")
    val_df.reset_index(inplace=True)
    val_df.rename(columns={"index": "ticker"}, inplace=True)
    return val_df