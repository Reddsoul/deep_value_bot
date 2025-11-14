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

from typing import List, Dict, Optional, Iterable  # add Iterable

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import time  # NEW
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

    # --- Universe selection ---
    # "sp_mid_small" -> S&P 400 + 600 mid/small caps (current behavior)
    # "nasdaq"       -> current NASDAQ ticker list via yfinance
    "UNIVERSE_SOURCE": "sp_mid_small",
}

_FUND_CACHE: Dict[str, Dict[str, object]] = {}

def get_nasdaq_universe(max_tickers: Optional[int] = None) -> List[str]:
    """
    Return a cleaned list of NASDAQ tickers suitable for yfinance.

    - Uses nasdaqtrader.com nasdaqtraded.txt
    - Drops:
        * Test issues
        * ETFs
        * Likely mutual funds (name contains "FUND")
        * Weird symbols that almost always break yfinance: contain ^, +, =, #
    - Converts BRK.A -> BRK-A, BF.B -> BF-B style tickers for class shares.

    Parameters
    ----------
    max_tickers : int or None
        If not None, truncate the list to the first `max_tickers` symbols.
        Useful for debugging.
    """
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    print("[Deep_Value_Bot] Loading NASDAQ universe from nasdaqtrader.com...")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to load NASDAQ tickers from nasdaqtrader.com: {e}")

    text = resp.text
    df = pd.read_csv(StringIO(text), sep="|")

    if "NASDAQ Symbol" not in df.columns:
        raise RuntimeError(
            "[Deep_Value_Bot] Unexpected format from nasdaqtrader.com "
            "(missing 'NASDAQ Symbol')."
        )

    # 1) Filter out test issues
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] == "N"]

    # 2) Filter out ETFs
    if "ETF" in df.columns:
        df = df[df["ETF"].astype(str).str.upper() != "Y"]

    # 3) Filter out likely mutual funds by name (heuristic)
    if "Security Name" in df.columns:
        name_series = df["Security Name"].astype(str)
        fund_mask = name_series.str.contains("FUND", case=False, na=False)
        df = df[~fund_mask]

    # Drop the 'File Creation Time' row if present
    sym_col = df["NASDAQ Symbol"].astype(str)
    mask_valid = sym_col.str.upper() != "FILE CREATION TIME"
    df = df[mask_valid]

    raw = (
        df["NASDAQ Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Drop obviously problematic characters for yfinance
    bad_chars = set("^+=#")
    filtered = [t for t in raw if not any(ch in t for ch in bad_chars)]

    # Convert class shares from BRK.A -> BRK-A, BF.B -> BF-B, etc.
    cleaned: List[str] = []
    for t in filtered:
        if "." in t:
            base, suffix = t.split(".", 1)
            # Heuristic: short suffix means it's probably a class/series
            if 1 <= len(suffix) <= 3:
                cleaned.append(f"{base}-{suffix}")
            else:
                cleaned.append(t)
        else:
            cleaned.append(t)

    tickers = sorted(set(cleaned))

    if max_tickers is not None:
        tickers = tickers[: int(max_tickers)]

    if not tickers:
        raise RuntimeError("[Deep_Value_Bot] NASDAQ universe is empty after filtering.")

    print(f"[Deep_Value_Bot] Loaded {len(tickers)} NASDAQ tickers after cleaning.")
    return tickers

def _chunked(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """Yield successive chunks from iterable of up to batch_size elements."""
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _safe_fetch_info(
    symbol: str,
    max_retries: int = 3,
    base_sleep: float = 1.0,
) -> Dict[str, object]:
    """
    Fetch yf.Ticker(symbol).info with retries + backoff.

    Returns an (possibly empty) dict on failure.
    """
    if symbol in _FUND_CACHE:
        return _FUND_CACHE[symbol]

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(symbol).info or {}
            _FUND_CACHE[symbol] = info
            return info
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt)
            print(
                f"[Deep_Value_Bot] yfinance info failed for {symbol} "
                f"(attempt {attempt + 1}/{max_retries}): {e}. "
                f"Sleeping {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)

    print(
        f"[Deep_Value_Bot] WARNING: giving up on fundamentals for {symbol} "
        f"after {max_retries} attempts. Last error: {last_err}"
    )
    _FUND_CACHE[symbol] = {}
    return {}

def _download_fundamentals(
    tickers: List[str],
    max_tickers: int = 1500,  # safety cap per run
) -> pd.DataFrame:
    """
    Download fundamental data for a list of tickers via yfinance.

    To avoid hammering Yahoo, we cap the number of tickers per run.
    """
    if not tickers:
        raise ValueError("_download_fundamentals: empty ticker list")

    uniq_all = sorted({str(t).strip().upper() for t in tickers if t})

    if len(uniq_all) > max_tickers:
        print(
            f"[Deep_Value_Bot] Fundamental universe {len(uniq_all)} > max_tickers={max_tickers}; "
            f"truncating for this run."
        )
        uniq = uniq_all[:max_tickers]
    else:
        uniq = uniq_all

    records: List[Dict[str, object]] = []

    print(
        f"[Deep_Value_Bot] Downloading fundamentals for "
        f"{len(uniq)} tickers via yfinance (with caching & retries)..."
    )

    for batch in _chunked(uniq, batch_size=64):
        for sym in batch:
            info = _safe_fetch_info(sym)
            if not info:
                record = {
                    "ticker": sym,
                    "marketCap": np.nan,
                    "totalAssets": np.nan,
                    "totalLiab": np.nan,
                    "netTangibleAssets": np.nan,
                    "cash": np.nan,
                    "totalDebt": np.nan,
                    "totalCurrentAssets": np.nan,
                    "totalCurrentLiabilities": np.nan,
                    "trailingEps": np.nan,
                }
            else:
                record = {
                    "ticker": sym,
                    "marketCap": info.get("marketCap", np.nan),
                    "totalAssets": info.get("totalAssets", np.nan),
                    "totalLiab": info.get("totalLiab", np.nan),
                    "netTangibleAssets": info.get("netTangibleAssets", np.nan),
                    "cash": info.get("cash", np.nan),
                    "totalDebt": info.get("totalDebt", np.nan),
                    "totalCurrentAssets": info.get("totalCurrentAssets", np.nan),
                    "totalCurrentLiabilities": info.get(
                        "totalCurrentLiabilities", np.nan
                    ),
                    "trailingEps": info.get("trailingEps", np.nan),
                }
            records.append(record)

        # Gentle pause between batches to be nice to Yahoo
        time.sleep(1.0)

    if not records:
        raise RuntimeError("No fundamental data could be fetched.")

    df = pd.DataFrame.from_records(records).set_index("ticker")

    # Ensure expected columns exist
    for col in [
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
        if col not in df.columns:
            df[col] = np.nan

    # Derived metrics (you already compute some later, but these are safe)
    mkt = df["marketCap"].astype(float)
    total_assets = df["totalAssets"].astype(float)
    total_liab = df["totalLiab"].astype(float)
    net_tang = df["netTangibleAssets"].astype(float)
    cash = df["cash"].astype(float)
    total_debt = df["totalDebt"].astype(float)
    ca = df["totalCurrentAssets"].astype(float)
    cl = df["totalCurrentLiabilities"].astype(float)

    df["p_tangible_book"] = np.where(net_tang > 0, mkt / net_tang, np.nan)
    ncav_num = ca - total_liab
    df["ncav_ratio"] = np.where(mkt > 0, ncav_num / mkt, np.nan)
    equity_approx = total_assets - total_liab
    df["debt_to_equity"] = np.where(equity_approx > 0, total_debt / equity_approx, np.nan)
    df["current_ratio"] = np.where(cl > 0, ca / cl, np.nan)

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
        If provided, use this universe instead of config-driven universe.
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

    # ---------------------------
    # Universe selection
    # ---------------------------
    if universe_override is not None:
        universe = list(universe_override)
    else:
        universe_source = cfg.get("UNIVERSE_SOURCE")
        if universe_source == "nasdaq":
            universe = get_nasdaq_universe()
        else:
            raise ValueError(
                f"Unsupported UNIVERSE_SOURCE={universe_source!r}. "
                "Use 'sp_mid_small' or 'nasdaq'."
            )

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