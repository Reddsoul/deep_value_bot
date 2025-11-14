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
FUNDAMENTAL_QUARANTINE: set[str] = set()  # NEW


def get_nasdaq_universe(max_tickers: Optional[int] = None) -> List[str]:
    """
    Return a deeply cleaned list of NASDAQ tickers suitable for yfinance.

    Cleaning stages:
      1) Drop test issues, ETFs.
      2) Drop funds (name contains 'FUND').
      3) Drop preferreds, warrants, rights, units, structured notes, SPAC-like names.
      4) Drop obviously broken symbols (^, +, =, #, /).
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

    total_raw = len(df)
    print(f"[Deep_Value_Bot] Raw NASDAQ rows: {total_raw}")

    # 1) Filter out test issues
    if "Test Issue" in df.columns:
        before = len(df)
        df = df[df["Test Issue"] == "N"]
        print(f"[Deep_Value_Bot] Removed {before - len(df)} test issues.")

    # 2) Filter out ETFs
    if "ETF" in df.columns:
        before = len(df)
        df = df[df["ETF"].astype(str).str.upper() != "Y"]
        print(f"[Deep_Value_Bot] Removed {before - len(df)} ETFs.")

    # 3) Filter out likely mutual funds by name
    if "Security Name" in df.columns:
        name_series = df["Security Name"].astype(str)
        before = len(df)
        fund_mask = name_series.str.contains("FUND", case=False, na=False)
        df = df[~fund_mask]
        print(f"[Deep_Value_Bot] Removed {before - len(df)} funds (name contains 'FUND').")

    # Drop the 'File Creation Time' row if present
    sym_col = df["NASDAQ Symbol"].astype(str)
    mask_valid = sym_col.str.upper() != "FILE CREATION TIME"
    df = df[mask_valid]

    # 4) Remove preferreds, warrants, rights, units, SPAC-like names, structured notes
    if "Security Name" in df.columns:
        name_series = df["Security Name"].astype(str).str.upper()
    else:
        name_series = pd.Series("", index=df.index)

    sym_series = df["NASDAQ Symbol"].astype(str).str.upper()

    # Preferreds: look at name tokens
    pref_mask = name_series.str.contains("PREFERRED", na=False) | name_series.str.contains("PFD", na=False)
    # Warrants
    warrant_mask = name_series.str.contains("WARRANT", na=False) | sym_series.str.endswith(("W", "WS", "WT"))
    # Rights
    rights_mask = name_series.str.contains("RIGHT", na=False) | sym_series.str.endswith("R")
    # Units
    units_mask = name_series.str.contains("UNIT", na=False) | sym_series.str.endswith("U")
    # Structured notes / weird instruments
    struct_mask = sym_series.str.contains("/", na=False)
    # SPAC-like (very heuristic)
    spac_mask = (
        name_series.str.contains("ACQUISITION", na=False)
        | name_series.str.contains("HOLDINGS", na=False)
        | name_series.str.contains("CAPITAL", na=False)
    )

    junk_mask = pref_mask | warrant_mask | rights_mask | units_mask | struct_mask | spac_mask
    before = len(df)
    df = df[~junk_mask]
    print(f"[Deep_Value_Bot] Removed {before - len(df)} preferreds/warrants/rights/units/SPAC-like/notes.")

    raw = (
        df["NASDAQ Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Drop obviously problematic characters for yfinance
    bad_chars = set("^+=#/")  # also drop '/' here as a last resort
    filtered = [t for t in raw if not any(ch in t for ch in bad_chars)]

    # Convert class shares from BRK.A -> BRK-A, BF.B -> BF-B, etc.
    cleaned: List[str] = []
    for t in filtered:
        if "." in t:
            base, suffix = t.split(".", 1)
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
        raise RuntimeError("[Deep_Value_Bot] NASDAQ universe is empty after cleaning.")

    print(f"[Deep_Value_Bot] Final cleaned NASDAQ universe: {len(tickers)} tickers.")
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
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> Dict[str, object]:
    """
    Fetch yf.Ticker(symbol).info with retries + backoff + fallbacks.

    - First try .info
    - On failure, fall back to .fast_info for marketCap.
    - On persistent failure, return a minimal dict and add to FUNDAMENTAL_QUARANTINE.
    """
    symbol = str(symbol).upper()

    if symbol in _FUND_CACHE:
        return _FUND_CACHE[symbol]

    if symbol in FUNDAMENTAL_QUARANTINE:
        return _FUND_CACHE.get(symbol, {})

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}
            if not info:
                # Try fast_info as a fallback source of marketCap
                try:
                    fi = getattr(t, "fast_info", None)
                    if fi is not None:
                        mcap = getattr(fi, "market_cap", None) or getattr(fi, "marketCap", None)
                        if mcap is not None:
                            info["marketCap"] = mcap
                except Exception:
                    pass

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
    FUNDAMENTAL_QUARANTINE.add(symbol)
    _FUND_CACHE[symbol] = {}
    return {}

def _download_fundamentals(
    tickers: List[str],
    max_tickers: Optional[int] = None,  # None = use full list
) -> pd.DataFrame:
    """
    Download fundamental data for a list of tickers via yfinance.

    To avoid hammering Yahoo, you *may* pass max_tickers to temporarily cap
    the number of tickers per run (for dev). In production, leave as None.
    """
    if not tickers:
        raise ValueError("_download_fundamentals: empty ticker list")

    uniq_all = sorted({str(t).strip().upper() for t in tickers if t})

    if max_tickers is not None and len(uniq_all) > max_tickers:
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
        f"{len(uniq)} tickers via yfinance (with caching, retries & fallbacks)..."
    )

    for batch in _chunked(uniq, batch_size=64):
        for sym in batch:
            info = _safe_fetch_info(sym)
            # Minimal robust defaults
            mcap = info.get("marketCap", np.nan)
            record = {
                "ticker": sym,
                "marketCap": mcap,
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

    # Drop obviously invalid rows (no marketCap at all)
    before = len(df)
    df = df[df["marketCap"].notna() & (df["marketCap"] > 0)]
    print(f"[Deep_Value_Bot] Dropped {before - len(df)} names with invalid marketCap.")

    # Derived metrics (keep in sync with _compute_value_metrics)
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

def _apply_graham_filters_with_relaxation(
    df: pd.DataFrame,
    cfg: Dict[str, object],
    as_of_date: pd.Timestamp,
    min_frac: float = 0.001,
    min_abs: int = 3,
) -> pd.DataFrame:
    """
    Apply Graham filters with adaptive relaxation so we never end up with
    zero (or near-zero) candidates silently.

    Steps:
      0) Strict filters (current cfg).
      1) Allow missing metrics, drop positive EPS requirement.
      2) Relax MoS thresholds slightly.
      3) If still too few, fall back to MoS-only and then best-score selection.
    """
    base = _apply_graham_filters(df, cfg)
    universe_size = len(base)
    if universe_size == 0:
        print("[Deep_Value_Bot] Graham filter: empty universe; returning empty DataFrame.")
        return base

    threshold = max(min_abs, int(np.ceil(min_frac * universe_size)))
    strict_pass = int(base["passes_all"].sum())
    print(
        f"[Deep_Value_Bot] {as_of_date.date()} | Strict Graham passes_all = "
        f"{strict_pass} / {universe_size} (min target {threshold})."
    )

    if strict_pass >= threshold:
        return base

    # Step 1: Allow missing metrics + drop positive EPS requirement
    cfg1 = dict(cfg)
    cfg1["ALLOW_MISSING_METRICS"] = True
    cfg1["REQUIRE_POSITIVE_EARNINGS_ENTRY"] = False
    relaxed1 = _apply_graham_filters(df, cfg1)
    pass1 = int(relaxed1["passes_all"].sum())
    print(
        f"[Deep_Value_Bot] Relaxation step 1 (allow missing, no EPS requirement): "
        f"{pass1} / {universe_size}."
    )
    if pass1 >= threshold:
        return relaxed1

    # Step 2: Relax MoS thresholds slightly
    cfg2 = dict(cfg1)
    if cfg2.get("MAX_P_TANGIBLE_BOOK_ENTRY") is not None:
        cfg2["MAX_P_TANGIBLE_BOOK_ENTRY"] = float(cfg2["MAX_P_TANGIBLE_BOOK_ENTRY"]) * 1.25
    if cfg2.get("MIN_NCAV_RATIO_ENTRY") is not None:
        cfg2["MIN_NCAV_RATIO_ENTRY"] = float(cfg2["MIN_NCAV_RATIO_ENTRY"]) * 0.8

    relaxed2 = _apply_graham_filters(df, cfg2)
    pass2 = int(relaxed2["passes_all"].sum())
    print(
        f"[Deep_Value_Bot] Relaxation step 2 (looser MoS): "
        f"{pass2} / {universe_size}."
    )
    if pass2 >= threshold:
        return relaxed2

    # Step 3: MoS-only and best-score fallback
    fallback = relaxed2.copy()
    if "mos_pass" in fallback.columns:
        fallback["passes_all"] = fallback["mos_pass"]
        pass3 = int(fallback["passes_all"].sum())
        print(
            f"[Deep_Value_Bot] Relaxation step 3 (MoS-only): "
            f"{pass3} / {universe_size}."
        )
    else:
        pass3 = 0

    if pass3 >= threshold:
        return fallback

    # Hard fallback: force at least N best-score names as candidates
    if "score" in fallback.columns:
        fallback_sorted = fallback.sort_values("score")
    else:
        fallback_sorted = fallback.copy()

    n_force = min(max(10, threshold), len(fallback_sorted))
    force_idx = fallback_sorted.index[:n_force]
    fallback.loc[:, "passes_all"] = False
    fallback.loc[force_idx, "passes_all"] = True
    print(
        f"[Deep_Value_Bot] Hard fallback: forcing top {n_force} by score as candidates "
        f"on {as_of_date.date()}."
    )

    return fallback

def run_screen(
    as_of_date,
    universe_override: Optional[List[str]] = None,
    mode: str = "filtered",
    config: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Run the deep value screen as of `as_of_date`.

    Guarantees:
      - Universe is cleaned.
      - Graham filters are applied with adaptive relaxation.
      - Never silently returns zero candidates in 'filtered' mode.
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
        universe_raw = [str(s).strip().upper() for s in universe_override if s]
        print(f"[Deep_Value_Bot] Using override universe of {len(universe_raw)} symbols.")
        universe = sorted(set(universe_raw))
    else:
        universe_source = cfg.get("UNIVERSE_SOURCE")
        if universe_source == "nasdaq":
            universe = get_nasdaq_universe()
        else:
            raise ValueError(
                f"Unsupported UNIVERSE_SOURCE={universe_source!r}. "
                "Use 'nasdaq' (S&P mid/small source not implemented here)."
            )

    if not universe:
        raise RuntimeError("Universe is empty; cannot run screen.")

    # Fundamentals + metrics
    fund_df = _download_fundamentals(universe, max_tickers=None)
    if fund_df.empty:
        raise RuntimeError("[Deep_Value_Bot] Fundamental DataFrame is empty after download.")

    val_df = _compute_value_metrics(fund_df)
    val_df = _apply_graham_filters_with_relaxation(val_df, cfg, as_of_date=as_of_date)

    # Transparency stats
    raw_count = len(val_df)
    mos_count = int(val_df.get("mos_pass", False).sum()) if "mos_pass" in val_df.columns else 0
    quality_count = int(val_df.get("passes_all", False).sum()) if "passes_all" in val_df.columns else 0
    print(
        f"[Deep_Value_Bot] {as_of_date.date()} | Raw universe: {raw_count} names | "
        f"MoS pass: {mos_count} | Final passes_all: {quality_count}"
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
        before = len(out)
        out = out[out["passes_all"]].copy()
        after = len(out)
        print(
            f"[Deep_Value_Bot] Filtered mode: {before} -> {after} names with passes_all=True."
        )

    # Rank by cheapness
    if "score" in out.columns:
        out.sort_values("score", inplace=True)

    out.reset_index(drop=True, inplace=True)
    return out