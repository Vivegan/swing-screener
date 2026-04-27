"""
Pre-breakout and breakout signal engine.

The intent: find stocks that are *coiling* (volatility contraction) inside an
uptrend, so we can position before the breakout fires — not after.

Pre-breakout score components:
  - NR7 (today's range is the narrowest in 7 days)
  - Bollinger Band squeeze (BB width near 6-month low)
  - Tight closes (stdev of last 5 closes near multi-month low)
  - Volume drying up (recent volume below longer avg)
  - Distance to pivot (close near recent N-day high)
  - Trend stack (price > 50 SMA > 200 SMA)
  - Relative strength vs SPY

Breakout firing:
  - Close above 20-day high
  - Volume > 1.5× 20-day avg
  - Close in upper third of day's range
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TickerSignals:
    ticker: str
    last_close: float
    last_high: float
    last_low: float
    last_volume: float
    avg_vol_20: float
    avg_vol_50: float
    pct_to_20d_high: float       # how far below 20d high (negative = below, 0 = at high)
    pct_to_50d_high: float
    high_20d: float              # 20-day high price
    high_5d: float               # 5-day high price (used as prebreakout buy-stop trigger)
    sma_20: float
    sma_50: float
    sma_200: float
    trend_stack: bool            # close > 50 SMA > 200 SMA
    bb_width_pct: float          # current BB width as % of price
    bb_squeeze: bool             # BB width in lowest 20% of last 6 months
    nr7: bool                    # today's range smallest in 7 days
    tight_closes: bool           # 5-day close stdev in lowest 25% of 6-month range
    volume_drying: bool          # 20d avg vol < 50d avg vol
    rs_vs_spy_3m: float          # 3-month return minus SPY 3-month return
    atr_14: float                # 14-day Average True Range (price units)
    adr_pct: float               # 20-day Average Daily Range as % of close
    vcp_score: int               # 0-3 — count of successive pullback contractions
    breakout_firing: bool        # close > 20d high, vol > 2.0×, upper third
    prebreakout_score: int       # 0-7, count of pre-breakout signals
    notes: list[str]


def compute_signals(df: pd.DataFrame, ticker: str, spy_returns: Optional[pd.Series] = None) -> Optional[TickerSignals]:
    """
    Compute signals from a daily OHLCV DataFrame. Returns None if data insufficient.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.copy().sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # Moving averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series(np.nan, index=close.index)

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid

    # 6-month lookback for "squeeze" comparison
    lookback = min(126, len(df) - 1)
    bb_width_recent = bb_width.iloc[-lookback:]
    bb_width_pct_now = float(bb_width.iloc[-1])
    if bb_width_recent.notna().sum() > 10:
        bb_squeeze = bb_width_pct_now <= np.nanpercentile(bb_width_recent.dropna(), 20)
    else:
        bb_squeeze = False

    # NR7 — today's range is the narrowest of the last 7 sessions
    daily_range = high - low
    nr7 = bool(daily_range.iloc[-1] == daily_range.iloc[-7:].min()) if len(df) >= 7 else False

    # Tight closes — stdev of last 5 closes is in the lowest 25% of last 6 months
    close_std_5 = close.rolling(5).std()
    if close_std_5.iloc[-lookback:].notna().sum() > 10:
        tight_closes = float(close_std_5.iloc[-1]) <= np.nanpercentile(
            close_std_5.iloc[-lookback:].dropna(), 25
        )
    else:
        tight_closes = False

    # Volume drying up
    avg_vol_20 = float(vol.rolling(20).mean().iloc[-1])
    avg_vol_50 = float(vol.rolling(50).mean().iloc[-1]) if len(df) >= 50 else avg_vol_20
    volume_drying = avg_vol_20 < avg_vol_50 * 0.95

    # Distance to recent highs
    high_20 = high.rolling(20).max().iloc[-1]
    high_50 = high.rolling(50).max().iloc[-1]
    last_close = float(close.iloc[-1])
    pct_to_20d_high = (last_close / high_20 - 1) * 100  # 0 means at high; negative = below
    pct_to_50d_high = (last_close / high_50 - 1) * 100

    # Trend stack
    s20 = float(sma_20.iloc[-1]) if pd.notna(sma_20.iloc[-1]) else np.nan
    s50 = float(sma_50.iloc[-1]) if pd.notna(sma_50.iloc[-1]) else np.nan
    s200 = float(sma_200.iloc[-1]) if pd.notna(sma_200.iloc[-1]) else np.nan
    trend_stack = bool(
        pd.notna(s50) and pd.notna(s200) and last_close > s50 > s200
    )

    # Relative strength vs SPY (3 months)
    rs_vs_spy_3m = 0.0
    if spy_returns is not None and len(df) >= 63:
        try:
            ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100
            spy_ret_3m = float(spy_returns.iloc[-1]) if not spy_returns.empty else 0.0
            rs_vs_spy_3m = float(ret_3m - spy_ret_3m)
        except Exception:
            rs_vs_spy_3m = 0.0

    # ATR(14) and ADR(20)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else float(tr.mean())
    adr_pct = float(((high - low) / close).rolling(20).mean().iloc[-1] * 100) if len(df) >= 20 else 0.0

    # Simplified VCP: detect successive contractions over the last ~30 sessions
    # Walk backwards through pivots and count how many times each pullback shrinks vs the prior.
    vcp_score = _vcp_score(close.iloc[-40:], lookahead=3)

    # 5-day high — used as prebreakout buy-stop trigger
    high_5d = float(high.iloc[-5:].max())

    # Breakout firing today (tightened: 2.0× vol)
    last_volume = float(vol.iloc[-1])
    today_range = float(high.iloc[-1] - low.iloc[-1])
    upper_third = (last_close >= float(low.iloc[-1]) + 0.66 * today_range) if today_range > 0 else False
    high_20_prior = float(high.iloc[-21:-1].max()) if len(df) >= 21 else high_20
    breakout_firing = bool(
        last_close > high_20_prior
        and last_volume > 2.0 * avg_vol_20
        and upper_third
        and trend_stack
    )

    # Pre-breakout score (0–7)
    notes = []
    score = 0
    if nr7:
        score += 1
        notes.append("NR7")
    if bb_squeeze:
        score += 1
        notes.append("BB squeeze")
    if tight_closes:
        score += 1
        notes.append("tight closes")
    if volume_drying:
        score += 1
        notes.append("vol drying")
    if pct_to_20d_high >= -3:  # within 3% of 20d high
        score += 1
        notes.append("near 20d high")
    if trend_stack:
        score += 1
        notes.append("trend stack")
    if rs_vs_spy_3m > 5:
        score += 1
        notes.append("RS+")

    return TickerSignals(
        ticker=ticker,
        last_close=last_close,
        last_high=float(high.iloc[-1]),
        last_low=float(low.iloc[-1]),
        last_volume=last_volume,
        avg_vol_20=avg_vol_20,
        avg_vol_50=avg_vol_50,
        pct_to_20d_high=float(pct_to_20d_high),
        pct_to_50d_high=float(pct_to_50d_high),
        high_20d=float(high_20),
        high_5d=high_5d,
        sma_20=s20 if pd.notna(s20) else 0.0,
        sma_50=s50 if pd.notna(s50) else 0.0,
        sma_200=s200 if pd.notna(s200) else 0.0,
        trend_stack=trend_stack,
        bb_width_pct=bb_width_pct_now,
        bb_squeeze=bb_squeeze,
        nr7=nr7,
        tight_closes=tight_closes,
        volume_drying=volume_drying,
        rs_vs_spy_3m=rs_vs_spy_3m,
        atr_14=atr_14,
        adr_pct=adr_pct,
        vcp_score=vcp_score,
        breakout_firing=breakout_firing,
        prebreakout_score=score,
        notes=notes,
    )


def _vcp_score(close: pd.Series, lookahead: int = 3) -> int:
    """
    Crude VCP detection: walk back through local highs and lows of the
    closing series, find pullbacks (drop from local high to next local low),
    and count how many *successive* pullbacks each shrink by ≥40% vs the prior one.

    Returns 0–3:
      0 = no contraction
      1 = one contraction (pullback B < A)
      2 = two successive contractions (C < B < A)
      3 = three successive (D < C < B < A) — Minervini-grade VCP
    """
    if close is None or len(close) < 15:
        return 0
    arr = close.dropna().values
    n = len(arr)
    pivots = []  # list of (index, price, kind) where kind = 'H' or 'L'

    for i in range(lookahead, n - lookahead):
        window = arr[i - lookahead : i + lookahead + 1]
        if arr[i] == window.max() and arr[i] != arr[i - 1]:
            pivots.append((i, arr[i], "H"))
        elif arr[i] == window.min() and arr[i] != arr[i - 1]:
            pivots.append((i, arr[i], "L"))

    # Build pullbacks as % drops from H -> next L
    pullbacks = []
    last_h = None
    for _, price, kind in pivots:
        if kind == "H":
            last_h = price
        elif kind == "L" and last_h is not None and price < last_h:
            pullbacks.append((last_h - price) / last_h)
            last_h = None

    # Count successive contractions
    score = 0
    for i in range(1, len(pullbacks)):
        if pullbacks[i] < 0.6 * pullbacks[i - 1]:  # each ≥40% smaller than prior
            score += 1
        else:
            score = 0  # break the chain — VCP requires *successive* contractions
    return min(score, 3)


def signals_to_row(s: TickerSignals) -> dict:
    return {
        "ticker": s.ticker,
        "close": round(s.last_close, 2),
        "prebreakout_score": s.prebreakout_score,
        "vcp_score": s.vcp_score,
        "breakout_firing": s.breakout_firing,
        "trend_stack": s.trend_stack,
        "pct_to_20d_high": round(s.pct_to_20d_high, 2),
        "pct_to_50d_high": round(s.pct_to_50d_high, 2),
        "rs_vs_spy_3m": round(s.rs_vs_spy_3m, 2),
        "nr7": s.nr7,
        "bb_squeeze": s.bb_squeeze,
        "tight_closes": s.tight_closes,
        "vol_drying": s.volume_drying,
        "avg_vol_20": int(s.avg_vol_20) if not np.isnan(s.avg_vol_20) else 0,
        "atr_14": round(s.atr_14, 2),
        "adr_pct": round(s.adr_pct, 2),
        "high_20d": round(s.high_20d, 2),
        "high_5d": round(s.high_5d, 2),
        "notes": ", ".join(s.notes),
    }
