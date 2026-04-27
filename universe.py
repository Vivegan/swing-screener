"""
Ticker universe loaders.

- S&P 500 from Wikipedia (live, cached daily)
- Curated swing list: high-momentum, liquid names that often live outside S&P 500
  or are S&P names with strong swing-trade history.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import io
import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Curated list of liquid, high-momentum tickers that swing well.
# Mix of mega-cap tech, AI, semis, growth, fintech, biotech, and a few cyclicals.
# Updated occasionally — feel free to edit.
CURATED_SWING = [
    # AI / semis
    "NVDA", "AMD", "AVGO", "MRVL", "ARM", "SMCI", "MU", "TSM", "ASML", "ALAB",
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX", "TSLA",
    # Growth / software
    "CRM", "NOW", "SNOW", "DDOG", "NET", "CRWD", "PLTR", "SHOP", "ABNB", "UBER",
    "MDB", "ZS", "PANW", "FTNT",
    # Fintech / payments
    "XYZ", "PYPL", "COIN", "HOOD", "AFRM", "SOFI", "V", "MA",  # XYZ = Block (formerly SQ)
    # China ADRs (volatile but tradable)
    "BABA", "PDD", "JD", "BIDU", "NIO", "LI", "XPEV",
    # Biotech / healthcare momentum
    "LLY", "VRTX", "REGN", "MRNA", "BNTX",
    # Energy / commodities (when in vogue)
    "OXY", "CVX", "FCX", "NEM",
    # Consumer / retail
    "LULU", "DECK", "RH", "CMG", "DKNG",
    # Industrials / defense
    "GE", "RTX", "LMT", "BA", "CAT",
    # Crypto-adjacent
    "MSTR", "MARA", "RIOT", "CLSK",
    # ETFs as macro context (not screened, but useful)
    "SPY", "QQQ", "IWM",
]

SP500_CACHE = CACHE_DIR / "sp500.csv"
CACHE_TTL_SECONDS = 24 * 3600  # refresh once a day


def _fetch_sp500_from_datahub() -> list[str]:
    """Fallback: datahub.io maintains a clean SP500 CSV."""
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; finnhub-screener/1.0)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def _fetch_sp500_from_wikipedia() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; finnhub-screener/1.0)"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    df = tables[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def load_sp500() -> list[str]:
    """Load S&P 500 tickers — Wikipedia first, datahub.io fallback, cached 24h."""
    if SP500_CACHE.exists() and (time.time() - SP500_CACHE.stat().st_mtime) < CACHE_TTL_SECONDS:
        df = pd.read_csv(SP500_CACHE)
        return df["Symbol"].tolist()

    last_err = None
    for fetcher in (_fetch_sp500_from_wikipedia, _fetch_sp500_from_datahub):
        try:
            df = fetcher()
            df.to_csv(SP500_CACHE, index=False)
            return df["Symbol"].tolist()
        except Exception as e:
            last_err = e
            print(f"[universe] {fetcher.__name__} failed: {e}")

    if SP500_CACHE.exists():
        print(f"[universe] All live fetchers failed; using stale cache.")
        df = pd.read_csv(SP500_CACHE)
        return df["Symbol"].tolist()
    raise RuntimeError(f"Failed to load S&P 500 and no cache available: {last_err}")


def load_curated_swing() -> list[str]:
    """Curated high-momentum swing-friendly tickers."""
    return [t for t in CURATED_SWING if t not in {"SPY", "QQQ", "IWM"}]


def load_universe(name: str = "all") -> list[str]:
    """
    Load a universe by name.

    Options:
      - "sp500"  — S&P 500 only
      - "swing"  — curated swing list only
      - "all"    — union of S&P 500 + curated (deduplicated)
    """
    name = name.lower()
    if name == "sp500":
        tickers = load_sp500()
    elif name == "swing":
        tickers = load_curated_swing()
    elif name == "all":
        tickers = sorted(set(load_sp500()) | set(load_curated_swing()))
    else:
        raise ValueError(f"Unknown universe: {name}")
    return tickers


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    tickers = load_universe(name)
    print(f"Universe '{name}': {len(tickers)} tickers")
    print(", ".join(tickers[:30]) + ("..." if len(tickers) > 30 else ""))
