# Finnhub Swing Screener

Breakout + pre-breakout swing screener for 1–10 day holds. Uses `yfinance` for OHLCV (Finnhub free tier doesn't expose US candles) and Finnhub for fundamentals, news, earnings calendar, and recommendation trends.

## Setup

```bash
cd ~/Desktop/Claude/Projects/Trading/finnhub-screener
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key (one-time):

```bash
echo 'export FINNHUB_API_KEY="your_key_here"' >> ~/.zshrc
source ~/.zshrc
```

Or use a `.env` file (copy from `.env.example`).

## Run

```bash
python screener.py                              # confluence (local + TV), full universe
python screener.py --source local               # local indicators only (fast)
python screener.py --source tv                  # TradingView screener only — full US universe
python screener.py --source confluence          # local hits validated by TradingView (recommended)

python screener.py --universe sp500             # S&P 500 only
python screener.py --universe swing             # Curated swing list only
python screener.py --top 25                     # Show top 25 by composite score
python screener.py --mode prebreakout           # Coiled setups (no trigger yet) — get in early
python screener.py --mode breakout              # Breakouts firing today
python screener.py --no-finnhub                 # Skip Finnhub enrichment (faster, no key needed)
```

**Three sources:**
- `local` — our universe + custom signals (yfinance OHLCV → NR7, BB squeeze, etc.)
- `tv` — TradingView's precomputed indicators across the full US large-cap universe
- `confluence` (default) — name must pass both local and TV filters; gets a big score bonus

**Trade plan parameters (sized for small accounts):**
```bash
python screener.py --account 1500 --risk-pct 0.01 --tradable-only
python screener.py --account 5000 --risk-pct 0.005 --stop-atr 2.0 --target-atr 3.0
```

Each row in the output now includes `entry`, `entry_type` (market vs stop_buy above pivot), `stop`, `target`, `rr`, `shares`, `dollars_at_risk`, `position_value`, and `time_stop_days`. `--tradable-only` filters to picks meeting the ADR floor (default 1.8% daily range) and minimum R:R (default 1.5).

Output goes to `output/watchlist_YYYY-MM-DD.csv` and `output/watchlist_YYYY-MM-DD.md` (or `watchlist_tv_*` for TV-only mode).

## What it screens for

**Pre-breakout (catching it before it fires):**
- NR7 — narrowest range in 7 days (volatility contraction)
- Bollinger Band squeeze — BB width at 6-month low
- Tight closes — stdev of last 5 closes near multi-month low
- Volume drying up — 20-day avg volume below 50-day avg
- Distance to pivot — close within 3% of 20/50-day high
- Trend stack — price > 50 SMA > 200 SMA
- Strong relative strength vs SPY
- VCP score (0–3) — successive pullback contractions (Minervini-style)

**Breakout (already firing):**
- Close above 20-day high
- Volume > 2.0× 20-day average (tightened for swing-trade quality)
- Close in upper third of day's range
- Trend stack confirmed

**Trade plan (per ticker):**
- Entry: market price (firing breakout) or stop-buy 0.5% above 5-day high (prebreakout trigger)
- Stop: entry − 1.5 × ATR(14)
- Target: entry + 2.5 × ATR(14)
- Position size: floor(account × risk% / risk-per-share), capped at 50% of account
- Time stop: exit if no progress in 5 trading days

**Risk filters (auto-flagged):**
- Earnings within 14 trading days (from Finnhub, chunked into 7-day windows due to free-tier limits)
- ADR (avg daily range) floor of 1.8% — names too sluggish for swing trading get filtered with `--tradable-only`
- Minimum R:R of 1.5 to be marked tradable
- Recommendation trend (analyst lean from Finnhub)

## Files

- `universe.py` — ticker lists (S&P 500 + curated swing list)
- `data.py` — OHLCV (yfinance) + Finnhub data fetch with caching
- `signals.py` — pre-breakout and breakout detection (NR7, BB squeeze, ATR, ADR, VCP)
- `tv_screener.py` — TradingView screener integration (3 preset queries)
- `trade_plan.py` — entry/stop/target/R:R/position size/time-stop calculator
- `screener.py` — main entry point, scoring, output
- `output/` — generated watchlists
- `cache/` — cached price / Finnhub / S&P data
