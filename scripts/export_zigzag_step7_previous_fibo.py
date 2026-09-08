from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.zigzag_mt5 import zigzag_mt5_with_point

DATA_PATH = REPO_ROOT / "data/eurusd/EURUSD_1m.csv"
RESULT_DIR = REPO_ROOT / "results/zigzag_step7"
POINT = 0.00001
DEPTH = 350
DEVIATION = 125
BACKSTEP = 40
FIB_RATIOS = (0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618, 3.618, 4.236)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    lower = {c.lower(): c for c in df.columns}
    time_col = next(lower[x] for x in ("timestamp", "datetime", "time", "date") if x in lower)
    cols = {lower[x]: x for x in ("open", "high", "low", "close") if x in lower}
    if len(cols) != 4:
        raise ValueError("Missing OHLC columns")
    df = df.rename(columns=cols)
    df.index = pd.to_datetime(df[time_col], errors="coerce")
    return df.loc[df.index.notna()].sort_index()[["open", "high", "low", "close"]]


def fib_price(start_price: float, end_price: float, ratio: float) -> float:
    lo, hi = sorted((start_price, end_price))
    rng = hi - lo
    if end_price > start_price:
        return hi - rng * ratio
    return lo + rng * ratio


def retracement_pct(price: float, prior_start: float, prior_end: float) -> float:
    """Unbounded retracement from the prior swing endpoint: 0% = no retrace, 100% = prior start, >100% = extension beyond prior start."""
    rng = abs(prior_end - prior_start)
    if rng == 0:
        return 0.0
    if prior_end > prior_start:  # prior UP, retrace downward
        return (prior_end - price) / rng * 100.0
    # prior DOWN, retrace upward
    return (price - prior_end) / rng * 100.0


def fib_zone(retr: float) -> str:
    if retr < 0:
        return "<0"
    levels = [23.6, 38.2, 50.0, 61.8, 78.6, 100.0, 127.2, 161.8, 200.0, 261.8, 361.8, 423.6]
    prev = 0.0
    for level in levels:
        if retr < level:
            return f"{prev:.1f}-{level:.1f}"
        prev = level
    return ">=423.6"


def main() -> None:
    data = load_data()
    zz = zigzag_mt5_with_point(
        data.high.to_numpy(), data.low.to_numpy(),
        depth=DEPTH, deviation_points=DEVIATION,
        backstep=BACKSTEP, point=POINT,
    )
    highs = data.high.to_numpy()
    lows = data.low.to_numpy()
    pivots = []
    for i, value in enumerate(zz):
        if not math.isfinite(float(value)):
            continue
        value = float(value)
        kind = "high" if abs(value - highs[i]) <= abs(value - lows[i]) else "low"
        pivots.append((i, data.index[i], value, kind))

    swings = []
    for n, (a, b) in enumerate(zip(pivots[:-1], pivots[1:]), start=1):
        a_bar, a_time, a_price, a_kind = a
        b_bar, b_time, b_price, b_kind = b
        swings.append({
            "swing_id": n,
            "start_bar": a_bar,
            "end_bar": b_bar,
            "start_time": a_time,
            "end_time": b_time,
            "start_kind": a_kind,
            "end_kind": b_kind,
            "start_price": a_price,
            "end_price": b_price,
            "direction": "UP" if b_price > a_price else "DOWN",
            "bars": b_bar - a_bar,
            "swing_pips": abs(b_price - a_price) * 10000.0,
        })

    rows = []
    point_rows = []
    for i, cur in enumerate(swings):
        row = dict(cur)
        if i == 0:
            rows.append(row)
            continue

        prior = swings[i - 1]
        row.update({
            "prior_swing_id": prior["swing_id"],
            "prior_direction": prior["direction"],
            "prior_swing_pips": prior["swing_pips"],
            "prior_bars": prior["bars"],
            "prior_start_price": prior["start_price"],
            "prior_end_price": prior["end_price"],
        })
        for r in FIB_RATIOS[1:]:
            row[f"prior_fib_{r*100:.1f}_price"] = fib_price(prior["start_price"], prior["end_price"], r)

        end_retr = retracement_pct(cur["end_price"], prior["start_price"], prior["end_price"])
        row["current_end_retracement_pct"] = end_retr
        row["current_end_fib_zone"] = fib_zone(end_retr)
        rows.append(row)

        # Every M1 point/bar inside the current swing is evaluated against the prior swing.
        # For an UP prior swing, the adverse/retracement side is the bar LOW; for a DOWN
        # prior swing, it is the bar HIGH. The endpoint/reversal is separately retained.
        for bar in range(cur["start_bar"], cur["end_bar"] + 1):
            if prior["direction"] == "UP":
                point_price = float(data.low.iloc[bar])
            else:
                point_price = float(data.high.iloc[bar])
            retr = retracement_pct(point_price, prior["start_price"], prior["end_price"])
            point_rows.append({
                "swing_id": cur["swing_id"],
                "prior_swing_id": prior["swing_id"],
                "timestamp": data.index[bar],
                "bar": bar,
                "prior_direction": prior["direction"],
                "point_price": point_price,
                "retracement_pct": retr,
                "fib_zone": fib_zone(retr),
                "is_swing_reversal": bar == cur["end_bar"],
            })

    out = pd.DataFrame(rows)
    points = pd.DataFrame(point_rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULT_DIR / "swings_with_previous_fibo.csv", index=False)
    points.to_csv(RESULT_DIR / "swing_points_previous_fibo.csv", index=False)

    valid = out.iloc[1:].copy() if len(out) > 1 else out.iloc[0:0].copy()
    if len(valid):
        zone_counts = valid["current_end_fib_zone"].value_counts().rename_axis("fib_zone").reset_index(name="count")
        zone_counts["pct"] = zone_counts["count"] / len(valid)
    else:
        zone_counts = pd.DataFrame(columns=["fib_zone", "count", "pct"])
    zone_counts.to_csv(RESULT_DIR / "previous_fibo_zone_distribution.csv", index=False)

    summary = pd.DataFrame([{
        "depth": DEPTH,
        "deviation_points": DEVIATION,
        "backstep": BACKSTEP,
        "point": POINT,
        "pivot_count": len(pivots),
        "swing_count": len(out),
        "fibo_eligible_swings": len(valid),
        "point_observations": len(points),
        "min_swing_pips": out["swing_pips"].min() if len(out) else math.nan,
        "max_swing_pips": out["swing_pips"].max() if len(out) else math.nan,
        "median_swing_pips": out["swing_pips"].median() if len(out) else math.nan,
        "min_prior_swing_pips": valid["prior_swing_pips"].min() if len(valid) else math.nan,
        "max_prior_swing_pips": valid["prior_swing_pips"].max() if len(valid) else math.nan,
        "median_prior_swing_pips": valid["prior_swing_pips"].median() if len(valid) else math.nan,
        "min_reversal_retracement_pct": valid["current_end_retracement_pct"].min() if len(valid) else math.nan,
        "max_reversal_retracement_pct": valid["current_end_retracement_pct"].max() if len(valid) else math.nan,
        "median_reversal_retracement_pct": valid["current_end_retracement_pct"].median() if len(valid) else math.nan,
        "mean_reversal_retracement_pct": valid["current_end_retracement_pct"].mean() if len(valid) else math.nan,
        "min_bars": out["bars"].min() if len(out) else math.nan,
        "max_bars": out["bars"].max() if len(out) else math.nan,
        "median_bars": out["bars"].median() if len(out) else math.nan,
    }])
    summary.to_csv(RESULT_DIR / "swing_previous_fibo_summary.csv", index=False)

    stats = []
    for direction in ("UP", "DOWN"):
        x = valid[valid["prior_direction"] == direction]
        stats.append({
            "prior_direction": direction,
            "count": len(x),
            "median_prior_swing_pips": x["prior_swing_pips"].median() if len(x) else math.nan,
            "median_current_swing_pips": x["swing_pips"].median() if len(x) else math.nan,
            "median_reversal_retracement_pct": x["current_end_retracement_pct"].median() if len(x) else math.nan,
            "mean_reversal_retracement_pct": x["current_end_retracement_pct"].mean() if len(x) else math.nan,
        })
    pd.DataFrame(stats).to_csv(RESULT_DIR / "previous_fibo_direction_stats.csv", index=False)

    # Use the exact unbounded reversal value for distribution: 0-23.6, ..., 78.6-100,
    # then extensions beyond 100 rather than clipping all extensions into >=100.
    ext_bins = [0, 23.6, 38.2, 50, 61.8, 78.6, 100, 127.2, 161.8, 200, 261.8, 361.8, 423.6, float("inf")]
    ext_labels = ["0-23.6", "23.6-38.2", "38.2-50", "50-61.8", "61.8-78.6", "78.6-100", "100-127.2", "127.2-161.8", "161.8-200", "200-261.8", "261.8-361.8", "361.8-423.6", ">=423.6"]
    if len(valid):
        dist = valid.assign(reversal_fib_bucket=pd.cut(valid["current_end_retracement_pct"], bins=ext_bins, labels=ext_labels, right=False)).groupby("reversal_fib_bucket", observed=False).size().reset_index(name="count")
        dist["pct"] = dist["count"] / len(valid)
    else:
        dist = pd.DataFrame(columns=["reversal_fib_bucket", "count", "pct"])
    dist.to_csv(RESULT_DIR / "reversal_fibo_distribution.csv", index=False)


if __name__ == "__main__":
    main()
