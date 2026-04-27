"""
Main screener entry point.

Usage:
  python screener.py                       # all (S&P + curated)
  python screener.py --universe sp500      # S&P 500 only
  python screener.py --universe swing      # curated swing list
  python screener.py --top 25              # show top N
  python screener.py --mode prebreakout    # only coiled, no trigger yet
  python screener.py --mode breakout       # only firing today
  python screener.py --no-finnhub          # skip Finnhub enrichment (faster)

Outputs:
  output/watchlist_YYYY-MM-DD.csv
  output/watchlist_YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from data import FinnhubClient, fetch_ohlcv_batch
from signals import compute_signals, signals_to_row
from trade_plan import TradePlanConfig, attach_trade_plans
from universe import load_universe

try:
    import tv_screener
    _HAS_TV = True
except Exception:
    _HAS_TV = False

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_spy_3m_return(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """Compute SPY's 3-month rolling return (%) for relative strength comparison."""
    spy = prices.get("SPY")
    if spy is None or spy.empty or len(spy) < 63:
        return pd.Series(dtype=float)
    return (spy["close"] / spy["close"].shift(63) - 1) * 100


def enrich_earnings_only(df: pd.DataFrame, fh: FinnhubClient | None, days_ahead: int = 14) -> pd.DataFrame:
    """Fast: one batch earnings_calendar call, applied to all tickers."""
    if fh is None or df.empty:
        df["earnings_soon"] = False
        df["earnings_date"] = ""
        return df
    cal = fh.earnings_calendar(days_ahead=days_ahead)
    earnings_map = {}
    if not cal.empty and "symbol" in cal.columns and "date" in cal.columns:
        for _, row in cal.iterrows():
            earnings_map[row["symbol"]] = row["date"]
    df["earnings_date"] = df["ticker"].map(lambda t: earnings_map.get(t, ""))
    df["earnings_soon"] = df["earnings_date"].astype(bool)
    return df


def enrich_recommendations(df: pd.DataFrame, fh: FinnhubClient | None) -> pd.DataFrame:
    """Slower: per-ticker recommendation_trend call. Apply only to a small set."""
    if fh is None or df.empty:
        df["reco_strong_buy"] = 0
        df["reco_buy"] = 0
        df["reco_hold"] = 0
        df["reco_sell"] = 0
        return df

    reco_sb, reco_b, reco_h, reco_s = [], [], [], []
    for ticker in df["ticker"]:
        reco = fh.recommendation_trend(ticker)
        if not reco.empty:
            latest = reco.iloc[0]
            reco_sb.append(int(latest.get("strongBuy", 0)))
            reco_b.append(int(latest.get("buy", 0)))
            reco_h.append(int(latest.get("hold", 0)))
            reco_s.append(int(latest.get("sell", 0)) + int(latest.get("strongSell", 0)))
        else:
            reco_sb.append(0)
            reco_b.append(0)
            reco_h.append(0)
            reco_s.append(0)
    df["reco_strong_buy"] = reco_sb
    df["reco_buy"] = reco_b
    df["reco_hold"] = reco_h
    df["reco_sell"] = reco_s
    return df


def composite_score(row: pd.Series) -> float:
    """
    Combined ranking score. Higher = better.

    - prebreakout_score (0-7) weighted heavily
    - breakout_firing bonus
    - relative strength bonus
    - TV confluence bonus (name confirmed by TradingView's screener too)
    - penalty for earnings within 7 days (don't want to be naked into a print)
    """
    score = float(row["prebreakout_score"]) * 10
    if row["breakout_firing"]:
        score += 15
    if row["trend_stack"]:
        score += 5
    score += max(0.0, min(20.0, float(row["rs_vs_spy_3m"]) * 0.5))
    if row.get("earnings_soon", False):
        score -= 12

    # TV confluence — passing both local and TV filters is a much stronger signal
    tv_mode = row.get("tv_mode", "")
    if tv_mode == "breakout":
        score += 20
    elif tv_mode == "prebreakout":
        score += 15
    elif tv_mode == "pullback":
        score += 8

    # Reward proximity to highs (negative pct_to_20d_high closer to 0 = better)
    pct = float(row["pct_to_20d_high"])
    if pct >= -1:
        score += 5
    elif pct >= -3:
        score += 3
    elif pct >= -5:
        score += 1
    return round(score, 2)


def enrich_with_tv(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate each ticker with which (if any) TradingView preset it appears in."""
    if not _HAS_TV or df.empty:
        df["tv_mode"] = ""
        df["tv_rsi"] = 0.0
        df["tv_adx"] = 0.0
        df["tv_perf_3m"] = 0.0
        df["tv_rel_vol_10d"] = 0.0
        df["tv_recommend"] = 0.0
        return df

    print("Running TradingView preset queries (breakout, prebreakout, pullback)...")
    try:
        tv_results = tv_screener.all_modes()
    except Exception as e:
        print(f"WARN: TradingView query failed ({e}); skipping TV enrichment.")
        df["tv_mode"] = ""
        df["tv_rsi"] = 0.0
        df["tv_adx"] = 0.0
        df["tv_perf_3m"] = 0.0
        df["tv_rel_vol_10d"] = 0.0
        df["tv_recommend"] = 0.0
        return df

    # Priority: breakout > prebreakout > pullback (a name in two buckets keeps the strongest)
    tv_lookup: dict[str, dict] = {}
    for mode in ["pullback", "prebreakout", "breakout"]:
        tv_df = tv_results.get(mode, pd.DataFrame())
        for k, v in tv_screener.tv_signals_to_dict(tv_df, mode).items():
            tv_lookup[k] = v  # later modes overwrite, so breakout wins

    df["tv_mode"] = df["ticker"].map(lambda t: tv_lookup.get(t, {}).get("tv_mode", ""))
    df["tv_rsi"] = df["ticker"].map(lambda t: tv_lookup.get(t, {}).get("tv_rsi", 0.0))
    df["tv_adx"] = df["ticker"].map(lambda t: tv_lookup.get(t, {}).get("tv_adx", 0.0))
    df["tv_perf_3m"] = df["ticker"].map(lambda t: tv_lookup.get(t, {}).get("tv_perf_3m", 0.0))
    df["tv_rel_vol_10d"] = df["ticker"].map(lambda t: tv_lookup.get(t, {}).get("tv_rel_vol_10d", 0.0))
    df["tv_recommend"] = df["ticker"].map(lambda t: tv_lookup.get(t, {}).get("tv_recommend", 0.0))

    n_confluence = (df["tv_mode"] != "").sum()
    print(f"  TV confluence: {n_confluence}/{len(df)} tickers also pass a TV preset")
    return df


def run_tv_only(args) -> pd.DataFrame:
    """Run TradingView-only mode (no local universe scan needed)."""
    if not _HAS_TV:
        raise RuntimeError("tradingview-screener not installed.")
    print("Running TradingView preset queries...")
    results = tv_screener.all_modes(
        min_market_cap=args.min_market_cap,
        min_price=args.min_price,
        min_avg_vol=args.min_avg_vol,
    )
    rows = []
    for mode in ["breakout", "prebreakout", "pullback"]:
        tv_df = results.get(mode, pd.DataFrame())
        if tv_df.empty:
            continue
        for _, r in tv_df.iterrows():
            rows.append(
                {
                    "ticker": r["ticker"],
                    "close": round(float(r["close"]), 2),
                    "tv_mode": mode,
                    "tv_rsi": round(float(r.get("RSI", 0)), 1),
                    "tv_adx": round(float(r.get("ADX", 0)), 1),
                    "tv_perf_3m": round(float(r.get("Perf.3M", 0)), 1),
                    "tv_rel_vol_10d": round(float(r.get("relative_volume_10d_calc", 0)), 2),
                    "tv_recommend": round(float(r.get("Recommend.All", 0)), 2),
                    "sector": r.get("sector", ""),
                    "pct_to_6m_high": round(float(r.get("pct_to_6m_high", 0) or 0), 2),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="first")
    # tv-only score: simple priority by mode + perf + rel vol
    mode_weight = {"breakout": 30, "prebreakout": 20, "pullback": 10}
    df["score"] = (
        df["tv_mode"].map(mode_weight).fillna(0)
        + df["tv_perf_3m"].clip(lower=0, upper=50) * 0.4
        + df["tv_rel_vol_10d"].clip(lower=0, upper=5) * 3
        + df["tv_recommend"].clip(lower=-1, upper=1) * 5
    ).round(2)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def render_markdown(df: pd.DataFrame, mode: str, universe_name: str, total_scanned: int) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [
        f"# Swing Watchlist — {today}",
        "",
        f"**Mode:** `{mode}` &nbsp;&nbsp; **Universe:** `{universe_name}` &nbsp;&nbsp; **Scanned:** {total_scanned} tickers &nbsp;&nbsp; **Hits:** {len(df)}",
        "",
    ]

    firing = df[df["breakout_firing"]].copy()
    coiled = df[(~df["breakout_firing"]) & (df["prebreakout_score"] >= 4)].copy()

    if mode in ("breakout", "all") and not firing.empty:
        lines += ["## Breakouts firing today", ""]
        lines += [_md_table(firing.head(15)), ""]

    if mode in ("prebreakout", "all") and not coiled.empty:
        lines += ["## Pre-breakout (coiled, waiting to trigger)", ""]
        lines += [_md_table(coiled.head(20)), ""]

    if mode == "all" and firing.empty and coiled.empty:
        lines += ["_No qualifying setups found today._", ""]

    lines += [
        "## Notes",
        "",
        "- `prebreakout_score` (0–7) counts NR7, BB squeeze, tight closes, volume drying, near-high, trend stack, relative strength.",
        "- `breakout_firing` = close above 20-day high on >1.5× volume in upper third of range.",
        "- `earnings_soon` = earnings within 7 trading days — caution.",
        "- Cross-reference with IBKR scanner for liquidity / option chain quality before sizing.",
        "",
    ]
    return "\n".join(lines)


def _md_table(df: pd.DataFrame) -> str:
    cols = [
        "ticker",
        "close",
        "score",
        "tradable",
        "entry",
        "entry_type",
        "stop",
        "target",
        "rr",
        "shares",
        "dollars_at_risk",
        "position_value",
        "adr_pct",
        "vcp_score",
        "prebreakout_score",
        "breakout_firing",
        "tv_mode",
        "earnings_soon",
        "earnings_date",
        "notes",
    ]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols]
    return sub.to_markdown(index=False)


def _render_tv_markdown(df: pd.DataFrame, mode: str) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [
        f"# TradingView Watchlist — {today}",
        "",
        f"**Source:** TradingView screener &nbsp;&nbsp; **Mode:** `{mode}` &nbsp;&nbsp; **Hits:** {len(df)}",
        "",
    ]
    cols = ["ticker", "close", "tv_mode", "score", "tv_rsi", "tv_adx", "tv_perf_3m", "tv_rel_vol_10d", "tv_recommend", "sector"]
    cols = [c for c in cols if c in df.columns]
    lines += [df[cols].to_markdown(index=False), ""]
    lines += [
        "## Notes",
        "",
        "- `tv_mode` = which TradingView preset matched: breakout / prebreakout / pullback.",
        "- `tv_recommend` = TradingView's all-indicator rating (-1 strong sell ... +1 strong buy).",
        "- These hits come from the *full US large-cap universe*, not just our curated list.",
        "",
    ]
    return "\n".join(lines)


def _env(name: str, default, cast=str):
    """Read from env with type casting, falling back to *default*."""
    val = os.environ.get(name)
    if val is None:
        return default
    return cast(val)


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Multi-source swing screener (local + TradingView + Finnhub)")
    parser.add_argument("--universe", default=_env("UNIVERSE", "all"), choices=["all", "sp500", "swing"])
    parser.add_argument("--mode", default=_env("MODE", "all"), choices=["all", "prebreakout", "breakout"])
    parser.add_argument(
        "--source",
        default=_env("SOURCE", "confluence"),
        choices=["local", "tv", "confluence"],
        help="local = our universe + indicators only; tv = TradingView screener only; confluence = local hits enriched with TV (recommended)",
    )
    parser.add_argument("--top", type=int, default=_env("TOP", 30, int))
    parser.add_argument("--no-finnhub", action="store_true", default=_env("NO_FINNHUB", False, lambda v: v.lower() in ("1", "true", "yes")), help="Skip Finnhub enrichment")
    parser.add_argument("--min-price", type=float, default=_env("MIN_PRICE", 5.0, float))
    parser.add_argument("--min-avg-vol", type=float, default=_env("MIN_AVG_VOL", 500_000, float))
    parser.add_argument("--min-market-cap", type=float, default=_env("MIN_MARKET_CAP", 2_000_000_000, float), help="(TV mode) min market cap")
    # Swing-trade trade-plan parameters
    parser.add_argument("--account", type=float, default=_env("ACCOUNT_SIZE", 1500.0, float), help="Account size in $ for position sizing")
    parser.add_argument("--risk-pct", type=float, default=_env("RISK_PCT", 0.01, float), help="Fraction of account to risk per trade (default 1%%)")
    parser.add_argument("--min-adr", type=float, default=_env("MIN_ADR_PCT", 1.8, float), help="Min average daily range %% for tradable picks")
    parser.add_argument("--min-rr", type=float, default=_env("MIN_RR", 1.5, float), help="Min reward:risk ratio for tradable picks")
    parser.add_argument("--stop-atr", type=float, default=_env("STOP_ATR_MULT", 1.5, float), help="Stop multiplier on ATR(14)")
    parser.add_argument("--target-atr", type=float, default=_env("TARGET_ATR_MULT", 2.5, float), help="Target multiplier on ATR(14)")
    parser.add_argument("--tradable-only", action="store_true", default=_env("TRADABLE_ONLY", False, lambda v: v.lower() in ("1", "true", "yes")), help="Show only picks meeting ADR + R:R floor")
    args = parser.parse_args()

    # ---- TV-only path: skip local OHLCV scan entirely ----
    if args.source == "tv":
        df = run_tv_only(args)
        if df.empty:
            print("No TradingView hits.")
            sys.exit(0)
        if args.mode == "breakout":
            df = df[df["tv_mode"] == "breakout"]
        elif args.mode == "prebreakout":
            df = df[df["tv_mode"] == "prebreakout"]
        df = df.head(args.top)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        csv_path = OUTPUT_DIR / f"watchlist_tv_{today}.csv"
        md_path = OUTPUT_DIR / f"watchlist_tv_{today}.md"
        df.to_csv(csv_path, index=False)
        md_path.write_text(_render_tv_markdown(df, args.mode))
        print()
        print(f"=== TV {args.mode} top {len(df)} ===")
        print(df.to_string(index=False))
        print(f"\nCSV  → {csv_path}\nMD   → {md_path}")
        return

    print(f"Loading universe '{args.universe}'...")
    tickers = load_universe(args.universe)
    # Always include SPY for relative strength baseline (won't appear in output)
    tickers_with_spy = sorted(set(tickers) | {"SPY"})
    print(f"Universe: {len(tickers)} tickers ({len(tickers_with_spy)} with SPY baseline)")

    print("Fetching OHLCV (cached when fresh)...")
    prices = fetch_ohlcv_batch(tickers_with_spy, period="1y")
    print(f"Got data for {len(prices)} tickers")

    spy_3m = compute_spy_3m_return(prices)

    print("Computing signals...")
    rows = []
    for ticker in tickers:
        df = prices.get(ticker)
        if df is None or df.empty:
            continue
        sig = compute_signals(df, ticker, spy_returns=spy_3m)
        if sig is None:
            continue
        if sig.last_close < args.min_price:
            continue
        if sig.avg_vol_20 < args.min_avg_vol:
            continue
        rows.append(signals_to_row(sig))

    if not rows:
        print("No tickers produced signals.")
        sys.exit(0)

    df = pd.DataFrame(rows)
    print(f"Computed signals for {len(df)} tickers")

    fh = None
    if not args.no_finnhub:
        try:
            fh = FinnhubClient()
            print("Enriching with Finnhub earnings calendar (1 call, all symbols)...")
        except RuntimeError as e:
            print(f"WARN: {e}. Continuing without Finnhub enrichment.")
            fh = None

    # Fast: one-shot earnings calendar for the whole universe (14-day window)
    df = enrich_earnings_only(df, fh, days_ahead=14)

    # TradingView confluence layer
    if args.source == "confluence":
        df = enrich_with_tv(df)
    else:
        df["tv_mode"] = ""

    df["score"] = df.apply(composite_score, axis=1)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Slow: recommendation_trend is per-ticker, so only enrich the shortlist
    if fh is not None:
        shortlist = df.head(min(args.top * 2, 50)).copy()
        print(f"Pulling Finnhub recommendation trends for top {len(shortlist)}...")
        shortlist = enrich_recommendations(shortlist, fh)
        # Merge back in
        for col in ["reco_strong_buy", "reco_buy", "reco_hold", "reco_sell"]:
            df[col] = 0
            df.loc[shortlist.index, col] = shortlist[col]
    else:
        df["reco_strong_buy"] = 0
        df["reco_buy"] = 0
        df["reco_hold"] = 0
        df["reco_sell"] = 0

    # Attach trade plans (entry/stop/target/size) for every row
    plan_cfg = TradePlanConfig(
        account_size=args.account,
        risk_pct=args.risk_pct,
        stop_atr_mult=args.stop_atr,
        target_atr_mult=args.target_atr,
        min_rr=args.min_rr,
        min_adr_pct=args.min_adr,
    )
    df = attach_trade_plans(df, plan_cfg)

    # Mode filtering
    if args.mode == "breakout":
        out = df[df["breakout_firing"]]
    elif args.mode == "prebreakout":
        out = df[(~df["breakout_firing"]) & (df["prebreakout_score"] >= 4)]
    else:
        out = df[(df["breakout_firing"]) | (df["prebreakout_score"] >= 4)]

    if args.tradable_only:
        out = out[out["tradable"]]

    out = out.head(args.top)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = OUTPUT_DIR / f"watchlist_{today}.csv"
    md_path = OUTPUT_DIR / f"watchlist_{today}.md"

    out.to_csv(csv_path, index=False)
    md = render_markdown(out, mode=args.mode, universe_name=args.universe, total_scanned=len(df))
    md_path.write_text(md)

    print()
    print(f"=== Top {len(out)} ({args.mode}, source={args.source}, account=${args.account:.0f}, risk={args.risk_pct*100:.1f}%) ===")
    print_cols = [
        "ticker", "close", "score", "tradable",
        "entry", "entry_type", "stop", "target", "rr", "shares",
        "dollars_at_risk", "position_value",
        "adr_pct", "atr_14", "vcp_score",
        "prebreakout_score", "breakout_firing", "tv_mode",
        "earnings_soon", "earnings_date",
        "rs_vs_spy_3m", "notes",
    ]
    print_cols = [c for c in print_cols if c in out.columns]
    print(out[print_cols].to_string(index=False))
    print()
    print(f"CSV  → {csv_path}")
    print(f"MD   → {md_path}")


if __name__ == "__main__":
    main()
