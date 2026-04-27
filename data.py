"""
Data layer.

OHLCV: yfinance (Finnhub free tier doesn't expose US stock candles).
Finnhub:
  - earnings calendar (risk filter)
  - recommendation trends (analyst direction)
  - company news + sentiment (catalyst awareness)

Disk cache for OHLCV (parquet) keeps re-runs fast.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

CACHE_DIR = Path(__file__).parent / "cache"
PRICE_CACHE = CACHE_DIR / "prices"
PRICE_CACHE.mkdir(parents=True, exist_ok=True)

PRICE_CACHE_TTL_HOURS = 6  # re-fetch prices if older than this


def _cache_path(ticker: str) -> Path:
    return PRICE_CACHE / f"{ticker}.parquet"


def _is_fresh(path: Path, hours: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < hours * 3600


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch daily OHLCV for one ticker. Returns empty DataFrame on failure.
    """
    cache = _cache_path(ticker)
    if _is_fresh(cache, PRICE_CACHE_TTL_HOURS):
        try:
            df = pd.read_parquet(cache)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return df
        # yfinance sometimes returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        df.index.name = "date"
        df.to_parquet(cache)
        return df
    except Exception as e:
        print(f"[data] {ticker} OHLCV fetch failed: {e}")
        return pd.DataFrame()


def fetch_ohlcv_batch(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """
    Bulk fetch — uses yfinance's batch downloader where possible, falls back per-ticker.
    """
    out: dict[str, pd.DataFrame] = {}
    to_fetch = []
    for t in tickers:
        cache = _cache_path(t)
        if _is_fresh(cache, PRICE_CACHE_TTL_HOURS):
            try:
                df = pd.read_parquet(cache)
                if not df.empty:
                    out[t] = df
                    continue
            except Exception:
                pass
        to_fetch.append(t)

    if not to_fetch:
        return out

    # Batch download in chunks of 50 to be polite
    CHUNK = 50
    for i in range(0, len(to_fetch), CHUNK):
        chunk = to_fetch[i : i + CHUNK]
        try:
            data = yf.download(
                chunk,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            for t in chunk:
                try:
                    if len(chunk) == 1:
                        df = data
                    else:
                        df = data[t]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    # Drop any duplicate columns (some tickers come back with repeats)
                    df = df.loc[:, ~df.columns.duplicated()]
                    df = df.rename(
                        columns={
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    ).dropna(how="all")
                    if df.empty:
                        continue
                    df.index.name = "date"
                    df.to_parquet(_cache_path(t))
                    out[t] = df
                except Exception as e:
                    print(f"[data] {t}: {e}")
        except Exception as e:
            print(f"[data] batch {i} failed ({e}); falling back per-ticker")
            for t in chunk:
                df = fetch_ohlcv(t, period=period)
                if not df.empty:
                    out[t] = df
    return out


# ---------------- Finnhub ----------------


class FinnhubClient:
    """Thin wrapper around finnhub-python with caching for slow-moving endpoints."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "FINNHUB_API_KEY not set. Add it to your environment or .env file."
            )
        import finnhub  # lazy import

        self.client = finnhub.Client(api_key=self.api_key)
        self.cache_dir = CACHE_DIR / "finnhub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cached(self, key: str, ttl_hours: int, fetcher):
        path = self.cache_dir / f"{key}.parquet"
        if _is_fresh(path, ttl_hours):
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        try:
            df = fetcher()
            if df is not None and not df.empty:
                df.to_parquet(path)
            return df
        except Exception as e:
            print(f"[finnhub] {key} failed: {e}")
            return pd.DataFrame()

    def earnings_calendar(self, days_ahead: int = 14) -> pd.DataFrame:
        """
        Upcoming earnings within the window. Used as risk filter.

        Finnhub's free tier truncates ranges >7 days, so we chunk into
        7-day windows and union the results.
        """
        chunks = []
        chunk_size = 7
        start = datetime.utcnow()
        days_remaining = days_ahead
        while days_remaining > 0:
            n = min(chunk_size, days_remaining)
            from_date = start.strftime("%Y-%m-%d")
            to_date = (start + timedelta(days=n)).strftime("%Y-%m-%d")

            def _fetch(_from=from_date, _to=to_date):
                res = self.client.earnings_calendar(_from=_from, to=_to, symbol="", international=False)
                rows = res.get("earningsCalendar", []) if isinstance(res, dict) else []
                return pd.DataFrame(rows)

            df_chunk = self._cached(
                f"earnings_{from_date}_{to_date}", ttl_hours=6, fetcher=_fetch
            )
            if df_chunk is not None and not df_chunk.empty:
                chunks.append(df_chunk)
            start = start + timedelta(days=n)
            days_remaining -= n

        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["symbol", "date"])
        return df

    def recommendation_trend(self, ticker: str) -> pd.DataFrame:
        def _fetch():
            res = self.client.recommendation_trends(ticker)
            return pd.DataFrame(res) if res else pd.DataFrame()

        return self._cached(f"reco_{ticker}", ttl_hours=24, fetcher=_fetch)

    def company_news(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")

        def _fetch():
            res = self.client.company_news(ticker, _from=from_date, to=to_date)
            return pd.DataFrame(res) if res else pd.DataFrame()

        return self._cached(f"news_{ticker}_{from_date}", ttl_hours=4, fetcher=_fetch)


if __name__ == "__main__":
    # Quick smoke test
    df = fetch_ohlcv("NVDA", period="6mo")
    print(f"NVDA: {len(df)} rows, last close = {df['close'].iloc[-1]:.2f}" if not df.empty else "NVDA: failed")

    try:
        fh = FinnhubClient()
        cal = fh.earnings_calendar(days_ahead=14)
        print(f"Earnings calendar (next 14d): {len(cal)} entries")
        reco = fh.recommendation_trend("NVDA")
        print(f"NVDA reco trend: {len(reco)} months")
    except RuntimeError as e:
        print(f"Finnhub not configured: {e}")
