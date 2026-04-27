"""
Trade plan: turns a screener signal into an executable trade.

For each row, computes:
  - entry            — current close (breakout firing) OR buy-stop above 5d high (prebreakout)
  - entry_type       — 'market' (already firing) or 'stop_buy' (waiting on trigger)
  - stop             — entry − stop_atr_mult × ATR(14)
  - target           — entry + target_atr_mult × ATR(14)
  - rr               — reward / risk ratio
  - shares           — floor(risk_dollars / (entry − stop)), risk_dollars = account × risk_pct
  - dollars_at_risk  — shares × (entry − stop)
  - position_value   — shares × entry
  - time_stop_days   — exit if no progress in N trading days

Also marks rows as `tradable` if they pass an ADR floor and meet a min R:R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class TradePlanConfig:
    account_size: float = 1500.0
    risk_pct: float = 0.01           # 1% of account per trade
    stop_atr_mult: float = 1.5       # stop = entry − 1.5 × ATR
    target_atr_mult: float = 2.5     # target = entry + 2.5 × ATR (1:1.67 R:R minimum)
    min_rr: float = 1.5              # require R:R ≥ 1.5 to mark tradable
    min_adr_pct: float = 1.8         # require avg daily range ≥ 1.8% of price
    time_stop_days: int = 5          # exit if no new high in N trading days
    prebreakout_buffer_pct: float = 0.005  # buy-stop placed 0.5% above 5d high
    max_position_pct: float = 0.5    # cap any single position at 50% of account


def compute_trade_plan(row: pd.Series, cfg: TradePlanConfig) -> dict:
    close = float(row.get("close", 0))
    atr = float(row.get("atr_14", 0))
    adr_pct = float(row.get("adr_pct", 0))
    high_5d = float(row.get("high_5d", close))
    high_20d = float(row.get("high_20d", close))
    breakout_firing = bool(row.get("breakout_firing", False))
    tv_mode = row.get("tv_mode", "")

    if atr <= 0 or close <= 0:
        return _empty_plan()

    # Entry
    if breakout_firing or tv_mode == "breakout":
        entry = close
        entry_type = "market"
        # Pivot for trailing context
        pivot = high_20d
    else:
        # Prebreakout: stop-buy slightly above the 5-day high (or 20d high if very close)
        trigger_base = max(high_5d, high_20d)
        entry = trigger_base * (1 + cfg.prebreakout_buffer_pct)
        entry_type = "stop_buy"
        pivot = trigger_base

    # Stop and target
    stop = entry - cfg.stop_atr_mult * atr
    target = entry + cfg.target_atr_mult * atr
    risk_per_share = entry - stop
    reward_per_share = target - entry
    rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

    # Position sizing
    risk_dollars = cfg.account_size * cfg.risk_pct
    shares = math.floor(risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    position_value = shares * entry
    # Cap by max position size
    max_position_value = cfg.account_size * cfg.max_position_pct
    if position_value > max_position_value and entry > 0:
        shares = math.floor(max_position_value / entry)
        position_value = shares * entry

    dollars_at_risk = shares * risk_per_share if shares > 0 else 0.0
    dollars_at_target = shares * reward_per_share if shares > 0 else 0.0

    tradable = (
        shares > 0
        and rr >= cfg.min_rr
        and adr_pct >= cfg.min_adr_pct
        and risk_per_share > 0
    )

    return {
        "entry": round(entry, 2),
        "entry_type": entry_type,
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round(rr, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
        "dollars_at_risk": round(dollars_at_risk, 2),
        "dollars_at_target": round(dollars_at_target, 2),
        "time_stop_days": cfg.time_stop_days,
        "pivot": round(pivot, 2),
        "tradable": tradable,
    }


def _empty_plan() -> dict:
    return {
        "entry": 0.0,
        "entry_type": "",
        "stop": 0.0,
        "target": 0.0,
        "rr": 0.0,
        "shares": 0,
        "position_value": 0.0,
        "dollars_at_risk": 0.0,
        "dollars_at_target": 0.0,
        "time_stop_days": 0,
        "pivot": 0.0,
        "tradable": False,
    }


def attach_trade_plans(df: pd.DataFrame, cfg: TradePlanConfig) -> pd.DataFrame:
    """Add trade plan columns to a DataFrame in-place (returns it)."""
    if df.empty:
        return df
    plans = df.apply(lambda r: compute_trade_plan(r, cfg), axis=1)
    plan_df = pd.DataFrame(plans.tolist(), index=df.index)
    return pd.concat([df, plan_df], axis=1)
