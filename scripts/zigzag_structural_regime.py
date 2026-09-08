from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_zigzag_temporal import load_data, metrics_for_period
from zigzag_mt5 import zigzag_mt5

RESULT_DIR = Path("results/zigzag_structural")
DATA_PATH = Path("data/eurusd/eurusd_m1.csv")
STEP3 = Path("results/zigzag_quarterly/quarter_top20.csv")


def pick_configs() -> list[tuple[int, int, int]]:
    df = pd.read_csv(STEP3)
    depths = sorted(df["depth"].dropna().astype(int).unique())
    deviations = sorted(df["deviation"].dropna().astype(int).unique())
    backsteps = sorted(df["backstep"].dropna().astype(int).unique())
    # Validate that Step 3 produced the expected robust neighborhood.
    if not depths or not deviations or not backsteps:
        raise RuntimeError("Step 3 candidate file is empty")
    center = int(round(float(df["depth"].median()) / 5) * 5)
    test_depths = sorted(set([center - 5, center, center + 5]))
    return [(d, v, b) for d in test_depths for v in deviations for b in backsteps]


def pivots_from_zigzag(data: pd.DataFrame, depth: int, deviation: int, backstep: int) -> pd.DataFrame:
    zz = zigzag_mt5(
        data["high"].to_numpy(),
        data["low"].to_numpy(),
        depth=depth,
        deviation=deviation,
        backstep=backstep,
    )
    rows = []
    for i, value in enumerate(zz):
        if not math.isfinite(float(value)):
            continue
        kind = "high" if abs(float(value) - float(data.iloc[i]["high"])) < abs(float(value) - float(data.iloc[i]["low"])) else "low"
        rows.append((i, data.index[i], float(value), kind))
    p = pd.DataFrame(rows, columns=["bar", "timestamp", "price", "kind"])
    if p.empty:
        return p
    # MT5 ZigZag alternates confirmed extrema. Keep the last occurrence if duplicates survive.
    p = p.sort_values("bar").reset_index(drop=True)
    return p


def structural_metrics(data: pd.DataFrame, pivots: pd.DataFrame) -> dict[str, float]:
    if len(pivots) < 4:
        return {k: float("nan") for k in [
            "pivots", "swings", "median_swing_pips", "median_bars",
            "trend_follow_rate", "reversal_follow_rate", "false_reversal_rate",
            "continuation_rate", "range_rate", "trend_rate", "range_efficiency",
            "trend_efficiency", "m5_alignment_rate",
        ]}

    swings = []
    for a, b in zip(pivots.iloc[:-1].itertuples(), pivots.iloc[1:].itertuples()):
        bars = int(b.bar - a.bar)
        move_pips = abs(b.price - a.price) * 10000.0
        if bars <= 0:
            continue
        # Path efficiency: net swing / intrabar absolute path over the same interval.
        lo = max(0, int(a.bar))
        hi = min(len(data) - 1, int(b.bar))
        close = data["close"].to_numpy()
        path = float(pd.Series(close[lo:hi + 1]).diff().abs().sum()) if hi > lo else 0.0
        efficiency = move_pips / (path * 10000.0) if path > 0 else 1.0
        direction = 1 if b.price > a.price else -1
        swings.append((a.bar, b.bar, bars, move_pips, efficiency, direction))

    s = pd.DataFrame(swings, columns=["a", "b", "bars", "swing_pips", "efficiency", "direction"])
    if s.empty:
        return {}

    # A structural trend is defined ex-post by two consecutive same-direction swings
    # with increasing swing magnitude. A reversal is the first opposite-direction swing.
    same = s["direction"].shift(1) == s["direction"]
    continuation_rate = float(same.iloc[1:].mean()) if len(s) > 1 else float("nan")
    reversal = ~same.iloc[1:] if len(s) > 1 else pd.Series(dtype=bool)
    false_reversal_rate = float((reversal & (s["swing_pips"].iloc[1:].to_numpy() < s["swing_pips"].shift(1).iloc[1:].to_numpy() * 0.5)).mean()) if len(s) > 1 else float("nan")

    # Regime labels use rolling 20-swing efficiency/dispersion; deliberately descriptive,
    # not a trading rule. Trend = directional persistence, range = low persistence.
    roll = s["direction"].rolling(10, min_periods=5).mean().abs()
    trend_mask = roll >= 0.5
    range_mask = roll < 0.5
    trend_rate = float(trend_mask.mean())
    range_rate = float(range_mask.mean())

    # M5 alignment: aggregate M1 closes into 5-minute blocks and compare the direction
    # of each completed ZigZag swing with the M5 move over the same endpoints.
    m5 = data[["close"]].copy()
    m5.index = pd.to_datetime(m5.index)
    m5c = m5["close"].resample("5min").last().dropna()
    aligned = []
    for r in s.itertuples():
        ts_a = pd.Timestamp(data.index[int(r.a)])
        ts_b = pd.Timestamp(data.index[int(r.b)])
        ca = m5c.asof(ts_a)
        cb = m5c.asof(ts_b)
        if pd.isna(ca) or pd.isna(cb) or cb == ca:
            continue
        aligned.append((1 if cb > ca else -1) == r.direction)
    m5_alignment = float(sum(aligned) / len(aligned)) if aligned else float("nan")

    return {
        "pivots": float(len(pivots)),
        "swings": float(len(s)),
        "median_swing_pips": float(s["swing_pips"].median()),
        "median_bars": float(s["bars"].median()),
        "trend_follow_rate": float(1.0 - false_reversal_rate),
        "reversal_follow_rate": float(reversal.mean()) if len(reversal) else float("nan"),
        "false_reversal_rate": false_reversal_rate,
        "continuation_rate": continuation_rate,
        "trend_rate": trend_rate,
        "range_rate": range_rate,
        "trend_efficiency": float(s.loc[trend_mask, "efficiency"].median()) if trend_mask.any() else float("nan"),
        "range_efficiency": float(s.loc[range_mask, "efficiency"].median()) if range_mask.any() else float("nan"),
        "m5_alignment_rate": m5_alignment,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    if not STEP3.exists():
        raise FileNotFoundError(STEP3)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(DATA_PATH)
    rows = []
    for depth, deviation, backstep in pick_configs():
        pivots = pivots_from_zigzag(data, depth, deviation, backstep)
        m = structural_metrics(data, pivots)
        rows.append({"depth": depth, "deviation": deviation, "backstep": backstep, **m})
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "structural_all_results.csv", index=False)

    numeric = [c for c in out.columns if c not in {"depth", "deviation", "backstep"}]
    summary = out.groupby("depth", as_index=False)[numeric].median()
    summary.to_csv(RESULT_DIR / "structural_depth_summary.csv", index=False)

    regime = out.groupby("depth", as_index=False)[[
        "trend_follow_rate", "reversal_follow_rate", "false_reversal_rate",
        "continuation_rate", "trend_rate", "range_rate", "trend_efficiency",
        "range_efficiency", "m5_alignment_rate",
    ]].median()
    regime.to_csv(RESULT_DIR / "regime_summary.csv", index=False)

    # Conservative validation flags: the structure must be materially present across
    # all tested parameter combinations, not only at one cell.
    robust = out.groupby("depth").agg(
        configs=("depth", "size"),
        median_false_reversal=("false_reversal_rate", "median"),
        max_false_reversal=("false_reversal_rate", "max"),
        median_m5_alignment=("m5_alignment_rate", "median"),
        min_m5_alignment=("m5_alignment_rate", "min"),
        median_continuation=("continuation_rate", "median"),
        median_trend_rate=("trend_rate", "median"),
        median_range_rate=("range_rate", "median"),
    ).reset_index()
    robust.to_csv(RESULT_DIR / "structural_robustness.csv", index=False)

    (RESULT_DIR / "README.md").write_text(
        """# Step 5 — ZigZag Structural Regime Validation\n\n"
        "Consumes Step 3 quarterly candidates and tests the robust Depth neighborhood around the Step 3 median.\n\n"
        "This step is descriptive validation only. It does not define an entry rule, optimize trading PnL, or select a final ZigZag setting.\n\n"
        "Metrics include swing geometry, directional continuation/reversal behavior, false-reversal proxy, trend/range regime share, efficiency by regime, and M1 swing direction versus completed M5 direction alignment.\n\n"
        "The output should be interpreted as structural evidence for whether ZigZag pivots carry stable market-regime information before entry-signal research begins.\n""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
