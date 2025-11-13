#!/usr/bin/env python3
"""
Deep_Value_Bot.py

Deep value screener that builds its universe from the S&P 400 + S&P 600
Wikipedia pages and pulls fundamentals from Yahoo Finance (via yfinance).

This version is refactored to be much more Graham-style:

- Explicit margin-of-safety thresholds (entry vs exit).
- Basic quality / financial-strength filters.
- Conservative handling of missing data.
- Transparent logging of how many names survive each filter stage.

Public API
----------
run_screen(as_of_date: pd.Timestamp | str,
           universe_override: list[str] | None = None,
           mode: str = "filtered",
           config: dict | None = None) -> pd.DataFrame

    Run the deep value screen as of (approximately) `as_of_date` and
    return a DataFrame of candidates with valuation metrics, filter flags
    and a `score` column where lower = cheaper.

    mode:
        - "filtered" (default): only rows where passes_all == True.
        - "raw": all rows, including mos_pass / quality_pass flags.

get_sp_universe(force_refresh: bool = False) -> list[str]
    Return the combined S&P 400 + S&P 600 ticker universe (as Yahoo tickers).
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO  # for pd.read_html on literal html strings

# ----------------------------------------------------------------------
# Graham-style configuration (edit these thresholds to tune behavior)
# ----------------------------------------------------------------------
GRAHAM_CONFIG: Dict[str, object] = {
    # --- Entry thresholds (margin of safety & quality) ---
    # Margin of safety: cheap vs book and NCAV
    "MAX_P_TANGIBLE_BOOK_ENTRY": 0.8,   # buy only if <= 0.8x tangible book
    "MIN_NCAV_RATIO_ENTRY": 0.5,        # NCAV / marketCap >= 0.5 (~ P <= 2 * NCAV)

    # Financial strength / liquidity
    "MIN_CURRENT_RATIO_ENTRY": 1.5,     # current assets / current liabilities
    "MAX_DEBT_TO_ASSETS_ENTRY": 0.5,    # totalDebt / totalAssets
    "MAX_DEBT_TO_MARKET_CAP_ENTRY": 1.0,  # totalDebt / marketCap

    # Earnings
    "REQUIRE_POSITIVE_EARNINGS_ENTRY": True,  # trailing EPS must be > 0

    # --- Exit thresholds (looser than entry, to avoid whipsaw) ---
    "MAX_P_TANGIBLE_BOOK_EXIT": 1.2,    # consider overvalued if > 1.2x TB
    "MIN_NCAV_RATIO_EXIT": 0.3,         # consider undervalued lost if < 0.3

    # --- Data handling behavior ---
    "ALLOW_MISSING_METRICS": False,
    "EXIT_ON_MISSING_FUNDAMENTALS": True,

    # Columns required to consider a name investable
    "REQUIRED_MOS_COLS": ["p_tangible_book", "ncav_ratio"],
    "REQUIRED_QUALITY_COLS": [
        "current_ratio",
        "debt_to_assets",
        "debt_to_equity_proxy",
        "trailingEps",
    ],
}


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

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError(f"No HTML tables found at {url}")

    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if any(c == "symbol" for c in cols):
            return tbl

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
    - marketCap
    - totalAssets
    - totalLiab
    - netTangibleAssets
    - cash
    - totalDebt
    - totalCurrentAssets
    - totalCurrentLiabilities
    - trailingEps
    (plus any other fields we might pull in future)
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

        rec: Dict[str, object] = {"ticker": t}
        # Core balance-sheet / valuation fields
        for key in [
            "marketCap",
            "totalAssets",
            "totalLiab",
            "netTangibleAssets",
            "cash",
            "totalDebt",
            "totalCurrentAssets",
            "totalCurrentLiabilities",
            "trailingEps",
        ]:
            rec[key] = info.get(key, np.nan)

        records.append(rec)

    if not records:
        raise RuntimeError("No fundamental data could be fetched.")

    df = pd.DataFrame.from_records(records).set_index("ticker")
    return df


def _compute_value_metrics(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute various deep value metrics, quality metrics and a combined score.
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

    # 1) Price-to-tangible-book (approx): market_cap / net_tangible_assets
    df["p_tangible_book"] = np.where(net_tang > 0, mkt / net_tang, np.nan)

    # 2) NCAV (Net Current Asset Value) = current assets - total liabilities
    ncav = ca - total_liab
    df["ncav"] = ncav
    df["ncav_ratio"] = np.where(mkt > 0, ncav / mkt, np.nan)

    # 3) Simple leverage proxy: total_debt / market_cap
    df["debt_to_equity_proxy"] = np.where(mkt > 0, total_debt / mkt, np.nan)

    # 4) Liquidity: current ratio
    df["current_ratio"] = np.where(cl > 0, ca / cl, np.nan)

    # 5) Debt to assets
    df["debt_to_assets"] = np.where(total_assets > 0, total_debt / total_assets, np.nan)

    # Combined "cheapness" score (lower = cheaper).
    score_components = []

    if "p_tangible_book" in df.columns:
        ptb = df["p_tangible_book"].clip(lower=0.0, upper=10.0)
        score_components.append(ptb)

    if "ncav_ratio" in df.columns:
        ncav_r = df["ncav_ratio"]
        ncav_score = (-ncav_r).clip(lower=-10.0, upper=10.0)  # flip sign
        score_components.append(ncav_score)

    if "debt_to_equity_proxy" in df.columns:
        lev = df["debt_to_equity_proxy"].clip(lower=0.0, upper=10.0)
        score_components.append(lev)

    if not score_components:
        raise RuntimeError("No score components were computed; check fundamentals.")

    score_mat = pd.concat(score_components, axis=1)
    df["score"] = score_mat.mean(axis=1)

    return df


def _apply_graham_filters(
    df: pd.DataFrame,
    cfg: Dict[str, object],
) -> pd.DataFrame:
    """
    Apply Graham-style margin-of-safety and quality filters.

    Adds:
        - mos_pass (bool)
        - quality_pass (bool)
        - passes_all (bool)
    """
    df = df.copy()

    mos_pass = pd.Series(True, index=df.index)
    quality_pass = pd.Series(True, index=df.index)

    # Missing-metric handling
    required_cols = set(cfg.get("REQUIRED_MOS_COLS", [])) | set(
        cfg.get("REQUIRED_QUALITY_COLS", [])
    )
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

    if not cfg.get("ALLOW_MISSING_METRICS", False) and required_cols:
        non_missing_mask = df[list(required_cols)].notna().all(axis=1)
    else:
        non_missing_mask = pd.Series(True, index=df.index)

    # Margin-of-safety filters
    max_ptb_entry = cfg.get("MAX_P_TANGIBLE_BOOK_ENTRY")
    if max_ptb_entry is not None:
        mos_pass &= df["p_tangible_book"] <= float(max_ptb_entry)

    min_ncav_ratio_entry = cfg.get("MIN_NCAV_RATIO_ENTRY")
    if min_ncav_ratio_entry is not None:
        mos_pass &= df["ncav_ratio"] >= float(min_ncav_ratio_entry)

    # Quality / financial strength filters
    min_current_ratio = cfg.get("MIN_CURRENT_RATIO_ENTRY")
    if min_current_ratio is not None:
        quality_pass &= df["current_ratio"] >= float(min_current_ratio)

    max_debt_assets = cfg.get("MAX_DEBT_TO_ASSETS_ENTRY")
    if max_debt_assets is not None:
        quality_pass &= df["debt_to_assets"] <= float(max_debt_assets)

    max_debt_mkt = cfg.get("MAX_DEBT_TO_MARKET_CAP_ENTRY")
    if max_debt_mkt is not None:
        quality_pass &= df["debt_to_equity_proxy"] <= float(max_debt_mkt)

    if cfg.get("REQUIRE_POSITIVE_EARNINGS_ENTRY", True):
        quality_pass &= df["trailingEps"].astype(float) > 0.0

    df["mos_pass"] = mos_pass & non_missing_mask
    df["quality_pass"] = quality_pass & non_missing_mask
    df["passes_all"] = df["mos_pass"] & df["quality_pass"]

    return df


def run_screen(
    as_of_date,
    universe_override: Optional[List[str]] = None,
    mode: str = "filtered",
    config: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Run the deep value screen as of `as_of_date`.

    Parameters
    ----------
    as_of_date : pd.Timestamp | str
    universe_override : list[str] or None
        If provided, use this universe instead of S&P 400/600.
    mode : {"filtered", "raw"}
        "raw"      -> return all names with mos/quality flags.
        "filtered" -> only names where passes_all == True.
    config : dict | None
        Optional override of GRAHAM_CONFIG keys.

    Returns
    -------
    pd.DataFrame
        Columns include at least:
            - ticker
            - symbol
            - marketCap, totalAssets, ...
            - p_tangible_book, ncav, ncav_ratio,
            - debt_to_equity_proxy, current_ratio, debt_to_assets
            - score
            - mos_pass, quality_pass, passes_all
    """
    if not isinstance(as_of_date, pd.Timestamp):
        as_of_date = pd.to_datetime(as_of_date)

    print(f"[Deep_Value_Bot] Running deep value screen as of {as_of_date.date()}...")

    # Configure thresholds
    cfg = dict(GRAHAM_CONFIG)
    if config:
        cfg.update(config)

    # Universe
    if universe_override is not None:
        universe = list(universe_override)
    else:
        universe = get_sp_universe()

    if not universe:
        raise RuntimeError("Universe is empty; cannot run screen.")

    # Fundamentals + metrics
    fund_df = _download_fundamentals(universe)
    val_df = _compute_value_metrics(fund_df)
    val_df = _apply_graham_filters(val_df, cfg)

    # Some transparency stats
    raw_count = len(val_df)
    mos_count = int(val_df["mos_pass"].sum())
    quality_count = int(val_df["passes_all"].sum())
    print(
        f"[Deep_Value_Bot] Raw universe: {raw_count} names | "
        f"MoS pass: {mos_count} | MoS+quality pass: {quality_count}"
    )

    # Reset index -> have ticker and symbol columns
    out = val_df.copy()
    out.reset_index(inplace=True)
    if "ticker" not in out.columns:
        out.rename(columns={"index": "ticker"}, inplace=True)
    out["ticker"] = out["ticker"].astype(str)
    if "symbol" not in out.columns:
        out["symbol"] = out["ticker"]

    # Apply mode
    if mode == "filtered":
        out = out[out["passes_all"]].copy()

    # Rank by cheapness
    if "score" in out.columns:
        out.sort_values("score", inplace=True)

    out.reset_index(drop=True, inplace=True)
    return out