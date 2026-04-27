"""
TradingView screener integration.

Why: TradingView precomputes hundreds of indicators (RSI, MACD, ADX, distance
to MAs, BB width, relative volume, performance windows) across the entire US
universe. Querying it is far faster than computing locally on thousands of
tickers, and it surfaces names *outside* our curated list.

We expose three preset queries aligned with our trading bias:

  - tv_breakout()    — close above 20d high on >1.5× rel vol, healthy trend
  - tv_prebreakout() — coiled setups: range-bound ADX, near 6M high, BB squeeze proxy
  - tv_pullback()    — pullback to 20/50 SMA in established uptrend, RSI not stretched

Note: tradingview-screener doesn't support column arithmetic in where clauses,
so we use simple comparisons on the server and apply % checks client-side.

No API key required — uses TradingView's public scanner endpoint.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from tradingview_screener import Query, col


# Common base columns we always want back
BASE_COLS = [
    "name",
    "close",
    "volume",
    "market_cap_basic",
    "exchange",
    "sector",
    "industry",
    "RSI",
    "ADX",
    "SMA20",
    "SMA50",
    "SMA200",
    "High.6M",
    "Low.6M",
    "price_52_week_high",
    "relative_volume_10d_calc",
    "average_volume_10d_calc",
    "Perf.W",
    "Perf.1M",
    "Perf.3M",
    "Recommend.All",
    "change",
    "BB.upper",
    "BB.lower",
]


def _base_filters(min_market_cap: float, min_price: float, min_avg_vol: float):
    """Liquidity / quality filters applied to every query."""
    return [
        col("exchange").isin(["NASDAQ", "NYSE"]),
        col("type") == "stock",
        col("market_cap_basic") > min_market_cap,
        col("close") > min_price,
        col("average_volume_10d_calc") > min_avg_vol,
    ]


def _scan(filters, order_col: str, order_asc: bool, limit: int) -> pd.DataFrame:
    count, df = (
        Query()
        .select(*BASE_COLS)
        .where(*filters)
        .order_by(order_col, ascending=order_asc)
        .limit(limit)
        .get_scanner_data()
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # TV returns both 'ticker' (NASDAQ:NVDA) and 'name' (NVDA). Keep name as the symbol.
    df = df.drop(columns=["ticker"], errors="ignore").rename(columns={"name": "ticker"})
    return df


def tv_breakout(
    limit: int = 100,
    min_market_cap: float = 2_000_000_000,
    min_price: float = 5.0,
    min_avg_vol: float = 750_000,
) -> pd.DataFrame:
    """
    Breakouts firing today.

    Server-side: trend stack + rel volume >= 1.5 + up day + ADX healthy.
    Client-side: close at or above 6-month high (true breakout vs bounce).
    """
    df = _scan(
        filters=[
            *_base_filters(min_market_cap, min_price, min_avg_vol),
            col("close") > col("SMA20"),
            col("close") > col("SMA50"),
            col("close") > col("SMA200"),
            col("relative_volume_10d_calc") >= 1.5,
            col("change") > 1.0,
            col("ADX") > 18,
        ],
        order_col="relative_volume_10d_calc",
        order_asc=False,
        limit=limit,
    )
    if df.empty:
        return df
    # client-side: within 2% of 6M high (allow names that just cleared a shorter pivot)
    df = df[df["close"] >= df["High.6M"] * 0.98].copy()
    df["pct_to_6m_high"] = (df["close"] / df["High.6M"] - 1) * 100
    return df.reset_index(drop=True)


def tv_prebreakout(
    limit: int = 200,
    min_market_cap: float = 2_000_000_000,
    min_price: float = 5.0,
    min_avg_vol: float = 750_000,
) -> pd.DataFrame:
    """
    Pre-breakout: coiled, near highs, no trigger yet.

    Server-side: trend stack + RSI healthy + 1M up + day flat-ish.
    Client-side: close within 4% of 6M high but not above it; BB width tight.
    """
    df = _scan(
        filters=[
            *_base_filters(min_market_cap, min_price, min_avg_vol),
            col("close") > col("SMA20"),
            col("SMA20") > col("SMA50"),
            col("SMA50") > col("SMA200"),
            col("RSI").between(45, 70),
            col("Perf.1M") > 0,
            col("change").between(-2.0, 2.0),
        ],
        order_col="Perf.3M",
        order_asc=False,
        limit=limit,
    )
    if df.empty:
        return df
    df["pct_to_6m_high"] = (df["close"] / df["High.6M"] - 1) * 100
    df["bb_width_pct"] = (df["BB.upper"] - df["BB.lower"]) / df["close"] * 100
    df = df[
        (df["pct_to_6m_high"] >= -6.0)
        & (df["pct_to_6m_high"] <= 1.0)
        & (df["bb_width_pct"] < 15.0)  # somewhat tight bands
    ].copy()
    return df.reset_index(drop=True)


def tv_pullback(
    limit: int = 200,
    min_market_cap: float = 2_000_000_000,
    min_price: float = 5.0,
    min_avg_vol: float = 750_000,
) -> pd.DataFrame:
    """
    Pullback to support in an uptrend.

    Server-side: trend stack, RSI cool, strong 3M perf, no panic vol.
    Client-side: close near SMA20 (within ±3%) and at/above SMA50 (within 3% above).
    """
    df = _scan(
        filters=[
            *_base_filters(min_market_cap, min_price, min_avg_vol),
            col("SMA20") > col("SMA50"),
            col("SMA50") > col("SMA200"),
            col("RSI").between(35, 55),
            col("Perf.3M") > 10,
            col("relative_volume_10d_calc") < 2.0,
        ],
        order_col="Perf.3M",
        order_asc=False,
        limit=limit,
    )
    if df.empty:
        return df
    df["pct_to_sma20"] = (df["close"] / df["SMA20"] - 1) * 100
    df["pct_to_sma50"] = (df["close"] / df["SMA50"] - 1) * 100
    df = df[
        (df["pct_to_sma20"].between(-3.0, 3.0))
        & (df["pct_to_sma50"] >= -1.0)
        & (df["pct_to_sma50"] <= 8.0)
    ].copy()
    return df.reset_index(drop=True)


def all_modes(
    min_market_cap: float = 2_000_000_000,
    min_price: float = 5.0,
    min_avg_vol: float = 750_000,
    limit_each: int = 200,
) -> dict[str, pd.DataFrame]:
    """Run all three preset queries and return labeled results."""
    return {
        "breakout": tv_breakout(limit=limit_each, min_market_cap=min_market_cap, min_price=min_price, min_avg_vol=min_avg_vol),
        "prebreakout": tv_prebreakout(limit=limit_each, min_market_cap=min_market_cap, min_price=min_price, min_avg_vol=min_avg_vol),
        "pullback": tv_pullback(limit=limit_each, min_market_cap=min_market_cap, min_price=min_price, min_avg_vol=min_avg_vol),
    }


def tv_signals_to_dict(df: pd.DataFrame, mode_label: str) -> dict[str, dict]:
    """Convert a TV result DataFrame into {ticker: {fields}} for join with local signals."""
    if df is None or df.empty:
        return {}
    out = {}
    for _, row in df.iterrows():
        out[row["ticker"]] = {
            "tv_mode": mode_label,
            "tv_rsi": float(row.get("RSI", 0) or 0),
            "tv_adx": float(row.get("ADX", 0) or 0),
            "tv_perf_3m": float(row.get("Perf.3M", 0) or 0),
            "tv_rel_vol_10d": float(row.get("relative_volume_10d_calc", 0) or 0),
            "tv_recommend": float(row.get("Recommend.All", 0) or 0),
            "tv_sector": row.get("sector", ""),
        }
    return out


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "breakout":
        df = tv_breakout()
        print(f"BREAKOUT: {len(df)} hits")
        print(df.head(20).to_string(index=False))
    elif mode == "prebreakout":
        df = tv_prebreakout()
        print(f"PREBREAKOUT: {len(df)} hits")
        print(df.head(20).to_string(index=False))
    elif mode == "pullback":
        df = tv_pullback()
        print(f"PULLBACK: {len(df)} hits")
        print(df.head(20).to_string(index=False))
    else:
        results = all_modes()
        for name, df in results.items():
            print(f"\n=== {name.upper()} ({len(df)} hits) ===")
            if not df.empty:
                cols = ["ticker", "close", "RSI", "ADX", "Perf.3M", "relative_volume_10d_calc", "sector"]
                cols = [c for c in cols if c in df.columns]
                print(df[cols].head(10).to_string(index=False))
