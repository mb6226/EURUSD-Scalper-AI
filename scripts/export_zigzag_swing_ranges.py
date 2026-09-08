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
RESULT_DIR = REPO_ROOT / "results/zigzag_step6"
POINT = 0.00001
DEPTH = 350
DEVIATION = 125
BACKSTEP = 40


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
        kind = "high" if abs(float(value) - highs[i]) <= abs(float(value) - lows[i]) else "low"
        pivots.append((i, data.index[i], float(value), kind))

    rows = []
    for n, (a, b) in enumerate(zip(pivots[:-1], pivots[1:]), start=1):
        a_bar, a_time, a_price, a_kind = a
        b_bar, b_time, b_price, b_kind = b
        bars = b_bar - a_bar
        pips = abs(b_price - a_price) * 10000.0
        rows.append({
            "swing_id": n,
            "start_time": a_time,
            "end_time": b_time,
            "start_kind": a_kind,
            "end_kind": b_kind,
            "start_price": a_price,
            "end_price": b_price,
            "direction": "UP" if b_price > a_price else "DOWN",
            "bars": bars,
            "swing_pips": pips,
        })

    swings = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    swings.to_csv(RESULT_DIR / "swings.csv", index=False)

    summary = pd.DataFrame([{
        "depth": DEPTH,
        "deviation_points": DEVIATION,
        "backstep": BACKSTEP,
        "point": POINT,
        "pivot_count": len(pivots),
        "swing_count": len(swings),
        "min_swing_pips": swings.swing_pips.min(),
        "max_swing_pips": swings.swing_pips.max(),
        "median_swing_pips": swings.swing_pips.median(),
        "min_swing_bars": swings.bars.min(),
        "max_swing_bars": swings.bars.max(),
        "median_swing_bars": swings.bars.median(),
    }])
    summary.to_csv(RESULT_DIR / "swing_min_max_summary.csv", index=False)

    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, float("inf")]
    labels = ["<10", "10-20", "20-30", "30-40", "40-50", "50-75", "75-100", "100-150", "150-200", "200-300", "300-500", "500-1000", ">=1000"]
    dist = swings.assign(swing_range= pd.cut(swings.swing_pips, bins=bins, labels=labels, right=False)).groupby("swing_range", observed=False).size().reset_index(name="count")
    dist["pct"] = dist["count"] / len(swings) if len(swings) else 0.0
    dist.to_csv(RESULT_DIR / "swing_size_distribution.csv", index=False)


if __name__ == "__main__":
    main()
