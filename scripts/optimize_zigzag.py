"""Run the 1,000-configuration MT5 ZigZag research sweep.

Outputs:
  results/zigzag/top20.csv
  results/zigzag/all_results.csv
  results/zigzag/stable_regions.csv
  results/zigzag/README.md
  results/zigzag/top20.json

The search grid is intentionally fixed to the research specification:
Depth 50..500 step 50, Deviation 25..250 step 25, Backstep 20..200 step 20.
Deviation is MT5 points; for 5-digit EURUSD, 10 points = 1 pip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

DEPTHS = np.arange(50, 501, 50, dtype=np.int64)
DEVIATIONS = np.arange(25, 251, 25, dtype=np.int64)
BACKSTEPS = np.arange(20, 201, 20, dtype=np.int64)


@njit(cache=True)
def _rolling_extreme(a: np.ndarray, depth: int, want_max: bool) -> np.ndarray:
    n = a.size
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    dq = np.empty(depth + 2, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        while head < tail and dq[head] <= i - depth:
            head += 1
        x = a[i]
        if want_max:
            while head < tail and a[dq[tail - 1]] <= x:
                tail -= 1
        else:
            while head < tail and a[dq[tail - 1]] >= x:
                tail -= 1
        dq[tail] = i
        tail += 1
        if i >= depth - 1:
            out[i] = a[dq[head]]
        # Compact occasionally so the fixed deque cannot drift indefinitely.
        if head > depth:
            k = head
            j = 0
            while k < tail:
                dq[j] = dq[k]
                j += 1
                k += 1
            tail = j
            head = 0
    return out


@njit(cache=True)
def evaluate_one(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    depth: int,
    deviation_points: int,
    backstep: int,
    point: float,
):
    """Return core quality metrics using the supplied MetaQuotes algorithm."""
    n = high.size
    hi_ext = _rolling_extreme(high, depth, True)
    lo_ext = _rolling_extreme(low, depth, False)
    hi_map = np.zeros(n, dtype=np.float64)
    lo_map = np.zeros(n, dtype=np.float64)
    last_low = 0.0
    last_high = 0.0
    start = depth - 1
    dev = deviation_points * point

    for shift in range(start, n):
        val = lo_ext[shift]
        if val == last_low:
            val = 0.0
        else:
            last_low = val
            if (low[shift] - val) > dev:
                val = 0.0
            else:
                first = max(start, shift - backstep)
                for j in range(shift - 1, first - 1, -1):
                    res = lo_map[j]
                    if res != 0.0 and res > val:
                        lo_map[j] = 0.0
        if low[shift] == val:
            lo_map[shift] = val

        val = hi_ext[shift]
        if val == last_high:
            val = 0.0
        else:
            last_high = val
            if (val - high[shift]) > dev:
                val = 0.0
            else:
                first = max(start, shift - backstep)
                for j in range(shift - 1, first - 1, -1):
                    res = hi_map[j]
                    if res != 0.0 and res < val:
                        hi_map[j] = 0.0
        if high[shift] == val:
            hi_map[shift] = val

    # We only retain the last pivot and metrics; this avoids allocating 1,000
    # full ZigZag arrays during the sweep.
    pivot_idx = np.empty(n, dtype=np.int64)
    pivot_px = np.empty(n, dtype=np.float64)
    pcount = 0
    search = 0
    last_low = 0.0
    last_high = 0.0
    last_low_pos = -1
    last_high_pos = -1

    for shift in range(start, n):
        if search == 0:
            if last_low == 0.0 and last_high == 0.0:
                if hi_map[shift] != 0.0:
                    last_high = high[shift]
                    last_high_pos = shift
                    search = -1
                    pivot_idx[pcount] = shift
                    pivot_px[pcount] = last_high
                    pcount += 1
                if lo_map[shift] != 0.0:
                    last_low = low[shift]
                    last_low_pos = shift
                    search = 1
                    pivot_idx[pcount] = shift
                    pivot_px[pcount] = last_low
                    pcount += 1
        elif search == 1:
            if lo_map[shift] != 0.0 and lo_map[shift] < last_low and hi_map[shift] == 0.0:
                if pcount > 0 and last_low_pos == pivot_idx[pcount - 1]:
                    pcount -= 1
                last_low_pos = shift
                last_low = lo_map[shift]
                pivot_idx[pcount] = shift
                pivot_px[pcount] = last_low
                pcount += 1
            if hi_map[shift] != 0.0 and lo_map[shift] == 0.0:
                last_high = hi_map[shift]
                last_high_pos = shift
                pivot_idx[pcount] = shift
                pivot_px[pcount] = last_high
                pcount += 1
                search = -1
        else:
            if hi_map[shift] != 0.0 and hi_map[shift] > last_high and lo_map[shift] == 0.0:
                if pcount > 0 and last_high_pos == pivot_idx[pcount - 1]:
                    pcount -= 1
                last_high_pos = shift
                last_high = hi_map[shift]
                pivot_idx[pcount] = shift
                pivot_px[pcount] = last_high
                pcount += 1
            if lo_map[shift] != 0.0 and hi_map[shift] == 0.0:
                last_low = lo_map[shift]
                last_low_pos = shift
                pivot_idx[pcount] = shift
                pivot_px[pcount] = last_low
                pcount += 1
                search = 1

    if pcount < 4:
        return (float(pcount), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Robust swing statistics and leg efficiency.
    swings = np.empty(pcount - 1, dtype=np.float64)
    bars = np.empty(pcount - 1, dtype=np.float64)
    eff = np.empty(pcount - 1, dtype=np.float64)
    for k in range(pcount - 1):
        i0 = pivot_idx[k]
        i1 = pivot_idx[k + 1]
        swings[k] = abs(pivot_px[k + 1] - pivot_px[k]) / (point * 10.0)
        bars[k] = i1 - i0
        path = 0.0
        for j in range(i0 + 1, i1 + 1):
            path += abs(close[j] - close[j - 1])
        net = abs(close[i1] - close[i0])
        eff[k] = net / path if path > 0.0 else 0.0

    swings_sorted = np.sort(swings)
    bars_sorted = np.sort(bars)
    eff_sorted = np.sort(eff)
    mid = swings_sorted.size // 2
    median_swing = swings_sorted[mid] if swings_sorted.size % 2 else 0.5 * (swings_sorted[mid - 1] + swings_sorted[mid])
    median_bars = bars_sorted[mid] if bars_sorted.size % 2 else 0.5 * (bars_sorted[mid - 1] + bars_sorted[mid])
    median_eff = eff_sorted[mid] if eff_sorted.size % 2 else 0.5 * (eff_sorted[mid - 1] + eff_sorted[mid])
    mean_eff = np.mean(eff)
    mean_swing = np.mean(swings)
    return (float(pcount), median_swing, mean_swing, median_bars, median_eff, mean_eff, float(swings.size))


@njit(parallel=True, cache=True)
def evaluate_grid(high, low, close, depths, deviations, backsteps, point):
    m = depths.size * deviations.size * backsteps.size
    out = np.empty((m, 10), dtype=np.float64)
    idx = 0
    for a in prange(depths.size):
        for b in range(deviations.size):
            for c in range(backsteps.size):
                d = int(depths[a])
                dv = int(deviations[b])
                bs = int(backsteps[c])
                vals = evaluate_one(high, low, close, d, dv, bs, point)
                out[idx, 0] = d
                out[idx, 1] = dv
                out[idx, 2] = bs
                out[idx, 3:] = vals
                idx += 1
    return out


def score_results(df: pd.DataFrame) -> pd.DataFrame:
    """Score configurations for clean major swings, not raw swing count."""
    x = df.copy()
    # Desirability transforms. Extremely dense ZigZags are penalized; very sparse
    # configurations are also penalized. The broad sweet spot is 100-1500 swings.
    density = x["pivot_count"].clip(lower=1)
    density_score = np.exp(-((np.log(density) - np.log(500.0)) ** 2) / (2 * 1.35**2))
    swing_score = np.clip(np.log1p(x["median_swing_pips"]) / np.log1p(25.0), 0, 1)
    efficiency_score = np.clip(x["median_efficiency"], 0, 1)
    spacing_score = np.exp(-((np.log1p(x["median_bars"]) - np.log1p(25.0)) ** 2) / (2 * 1.25**2))
    x["score"] = 100.0 * (
        0.35 * efficiency_score
        + 0.30 * swing_score
        + 0.20 * density_score
        + 0.15 * spacing_score
    )
    return x.sort_values("score", ascending=False).reset_index(drop=True)


def stable_regions(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Find connected-ish parameter neighborhoods around strong, non-isolated scores."""
    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)
    top = ranked.head(top_n).copy()
    rows = []
    for _, r in top.iterrows():
        mask = (
            (abs(df.depth - r.depth) <= 50)
            & (abs(df.deviation - r.deviation) <= 25)
            & (abs(df.backstep - r.backstep) <= 20)
        )
        neigh = df.loc[mask, "score"]
        if len(neigh) < 5:
            continue
        rows.append({
            "depth": int(r.depth),
            "deviation": int(r.deviation),
            "backstep": int(r.backstep),
            "score": float(r.score),
            "neighborhood_n": int(len(neigh)),
            "neighborhood_mean": float(neigh.mean()),
            "neighborhood_median": float(neigh.median()),
            "neighborhood_min": float(neigh.min()),
            "neighborhood_max": float(neigh.max()),
            "stability": float(neigh.mean() / max(r.score, 1e-9)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["stability", "neighborhood_mean", "score"], ascending=False).drop_duplicates(
        subset=["depth", "deviation", "backstep"]
    ).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/eurusd/EURUSD_1m.parquet")
    ap.add_argument("--output-dir", default="results/zigzag")
    ap.add_argument("--point", type=float, default=1e-5, help="EURUSD MT5 point size; 1e-5 for 5-digit pricing")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    high = df.high.to_numpy(np.float64)
    low = df.low.to_numpy(np.float64)
    close = df.close.to_numpy(np.float64)

    expected = len(DEPTHS) * len(DEVIATIONS) * len(BACKSTEPS)
    print(f"Rows: {len(df):,}; configurations: {expected:,}; point={args.point:g}")
    raw = evaluate_grid(high, low, close, DEPTHS, DEVIATIONS, BACKSTEPS, args.point)
    cols = ["depth", "deviation", "backstep", "pivot_count", "median_swing_pips", "mean_swing_pips", "median_bars", "median_efficiency", "mean_efficiency", "swing_count"]
    results = pd.DataFrame(raw, columns=cols)
    results = score_results(results)

    # Parameter-grid stability is computed on the full 1,000 rows.
    stable = stable_regions(results)
    results["deviation_pips"] = results.deviation / 10.0
    stable["deviation_pips"] = stable.deviation / 10.0 if not stable.empty else stable.get("deviation", pd.Series(dtype=float))

    results.to_csv(outdir / "all_results.csv", index=False, float_format="%.6f")
    results.head(20).to_csv(outdir / "top20.csv", index=False, float_format="%.6f")
    stable.head(20).to_csv(outdir / "stable_regions.csv", index=False, float_format="%.6f")
    results.head(20).to_json(outdir / "top20.json", orient="records", indent=2)

    best = results.iloc[0]
    md = f"""# EURUSD M1 MT5 ZigZag Optimization\n\nGenerated from `{args.input}` with the supplied MetaQuotes ZigZag semantics.\n\n- Configurations: **{expected:,}**\n- Rows: **{len(df):,}**\n- Point size: `{args.point:g}`\n- Deviation is **MT5 points**; on 5-digit EURUSD, 10 points = 1 pip.\n- Objective: clean major swings, leg efficiency, useful swing scale, and non-pathological swing density.\n- `stable_regions.csv` ranks neighborhoods where nearby parameter values remain strong; this is preferred to an isolated single best cell.\n\n## Best configuration\n\n- Depth: **{int(best.depth)}**\n- Deviation: **{int(best.deviation)} points ({best.deviation_pips:.1f} pips)**\n- Backstep: **{int(best.backstep)}**\n- Score: **{best.score:.3f}**\n- Pivot count: **{int(best.pivot_count)}**\n- Median swing: **{best.median_swing_pips:.3f} pips**\n- Median leg efficiency: **{best.median_efficiency:.3f}**\n\nThe score is a research ranking, not a claim of future trading performance.\n"""
    (outdir / "README.md").write_text(md, encoding="utf-8")
    print(md)
    print("Top 20:")
    print(results.head(20).to_string(index=False))
    print("Stable regions:")
    print(stable.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
