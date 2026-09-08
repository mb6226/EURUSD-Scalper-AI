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
FIB_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786, 1.000)


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


def fib_prices(start_price: float, end_price: float) -> dict[str, float]:
    lo, hi = sorted((start_price, end_price))
    rng = hi - lo
    out: dict[str, float] = {}
    if end_price > start_price:
        # Previous swing was UP: retracement is measured down from its high.
        for r in FIB_RATIOS:
            out[f"fib_{r*100:.1f}_price"] = hi - rng * r
    else:
        # Previous swing was DOWN: retracement is measured up from its low.
        for r in FIB_RATIOS:
            out[f"fib_{r*100:.1f}_price"] = lo + rng * r
    return out


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
    for i, cur in enumerate(swings):
        row = dict(cur)
        if i == 0:
            row.update({
                "prior_swing_id": pd.NA,
                "prior_direction": pd.NA,
                "prior_swing_pips": pd.NA,
                "prior_bars": pd.NA,
                "prior_start_price": pd.NA,
                "prior_end_price": pd.NA,
                "prior_fib_23.6_price": pd.NA,
                "prior_fib_38.2_price": pd.NA,
                "prior_fib_50.0_price": pd.NA,
                "prior_fib_61.8_price": pd.NA,
                "prior_fib_78.6_price": pd.NA,
                "prior_fib_100.0_price": pd.NA,
                "current_end_vs_prior_range_pct": pd.NA,
                "current_end_vs_prior_236_pct": pd.NA,
                "current_end_vs_prior_382_pct": pd.NA,
                "current_end_vs_prior_500_pct": pd.NA,
                "current_end_vs_prior_618_pct": pd.NA,
                "current_end_vs_prior_786_pct": pd.NA,
                "current_end_vs_prior_1000_pct": pd.NA,
                "current_end_fib_zone": pd.NA,
            })
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
        fps = fib_prices(prior["start_price"], prior["end_price"])
        row.update({f"prior_{k}": v for k, v in fps.items()})

        prior_lo, prior_hi = sorted((prior["start_price"], prior["end_price"]))
        prior_rng = prior_hi - prior_lo
        end = cur["end_price"]
        row["current_end_vs_prior_range_pct"] = ((end - prior_lo) / prior_rng * 100.0) if prior_rng else 0.0

        # Express current endpoint in the same retracement convention as the prior swing.
        if prior["direction"] == "UP":
            row["current_end_vs_prior_236_pct"] = (prior["end_price"] - end) / prior_rng * 100.0 if prior_rng else 0.0
        else:
            row["current_end_vs_prior_236_pct"] = (end - prior["end_price"]) / prior_rng * 100.0 if prior_rng else 0.0
        # The values above are a normalized retracement percentage; named copies make
        # the dataset explicit for downstream research without pretending they are prices.
        for pct in (38.2, 50.0, 61.8, 78.6, 100.0):
            row[f"current_end_vs_prior_{pct:.1f}_pct"] = row["current_end_vs_prior_236_pct"]

        retr = row["current_end_vs_prior_236_pct"]
        if retr < 23.6:
            zone = "<23.6"
        elif retr < 38.2:
            zone = "23.6-38.2"
        elif retr < 50.0:
            zone = "38.2-50.0"
        elif retr < 61.8:
            zone = "50.0-61.8"
        elif retr < 78.6:
            zone = "61.8-78.6"
        elif retr < 100.0:
            zone = "78.6-100.0"
        else:
            zone = ">=100.0"
        row["current_end_fib_zone"] = zone
        rows.append(row)

    out = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULT_DIR / "swings_with_previous_fibo.csv", index=False)

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
        "min_swing_pips": out["swing_pips"].min() if len(out) else math.nan,
        "max_swing_pips": out["swing_pips"].max() if len(out) else math.nan,
        "median_swing_pips": out["swing_pips"].median() if len(out) else math.nan,
        "min_prior_swing_pips": valid["prior_swing_pips"].min() if len(valid) else math.nan,
        "max_prior_swing_pips": valid["prior_swing_pips"].max() if len(valid) else math.nan,
        "median_prior_swing_pips": valid["prior_swing_pips"].median() if len(valid) else math.nan,
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
            "median_current_end_retracement_pct": x["current_end_vs_prior_236_pct"].median() if len(x) else math.nan,
            "mean_current_end_retracement_pct": x["current_end_vs_prior_236_pct"].mean() if len(x) else math.nan,
        })
    pd.DataFrame(stats).to_csv(RESULT_DIR / "previous_fibo_direction_stats.csv", index=False)


if __name__ == "__main__":
    main()
