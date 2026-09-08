from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.evaluate_zigzag_temporal import metrics_for_period


def inherited_region(path: Path) -> tuple[int, list[int], list[int]]:
    s = pd.read_csv(path)
    required = {"depth", "deviation", "backstep"}
    missing = required - set(s.columns)
    if missing:
        raise ValueError(f"Step 3 results missing columns: {sorted(missing)}")
    top = s.head(20).copy()
    center_depth = int(round(float(top.depth.median()) / 5.0) * 5)
    deviations = sorted({int(x) for x in top.deviation})
    backsteps = sorted({int(x) for x in top.backstep})
    return center_depth, deviations, backsteps


def build_candidates(step3_path: Path) -> pd.DataFrame:
    center, inherited_devs, inherited_backsteps = inherited_region(step3_path)
    depths = list(range(center - 50, center + 51, 5))
    # Step 3 showed the deviation range itself is stable; keep that range intact.
    deviations = inherited_devs
    # Probe one neighboring backstep beyond the Step 3 stable pair without exploding the grid.
    bmax = max(inherited_backsteps)
    backsteps = sorted(set(inherited_backsteps + [bmax + 20]))
    rows = [(depth, deviation, backstep)
            for depth in depths for deviation in deviations for backstep in backsteps]
    return pd.DataFrame(rows, columns=["depth", "deviation", "backstep"])


def score(row: pd.Series, baseline: pd.Series) -> float:
    swing_ratio = min(row.median_swing_pips, baseline.median_swing_pips) / max(row.median_swing_pips, baseline.median_swing_pips)
    bars_ratio = min(row.median_bars, baseline.median_bars) / max(row.median_bars, baseline.median_bars)
    eff_ratio = min(row.median_efficiency, baseline.median_efficiency) / max(row.median_efficiency, baseline.median_efficiency)
    pivot_ratio = min(row.pivot_count, baseline.pivot_count) / max(row.pivot_count, baseline.pivot_count)
    return float(0.30 * eff_ratio + 0.25 * swing_ratio + 0.20 * bars_ratio + 0.25 * pivot_ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/eurusd/EURUSD_1m.parquet")
    ap.add_argument("--step3", default="results/zigzag_quarterly/quarter_top20.csv")
    ap.add_argument("--output-dir", default="results/zigzag_robustness")
    ap.add_argument("--point", type=float, default=1e-5)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(args.input).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    candidates = build_candidates(Path(args.step3))
    candidates.to_csv(out / "robustness_candidates.csv", index=False)

    step3 = pd.read_csv(args.step3).head(20)
    baseline_cfg = {
        "depth": int(round(float(step3.depth.median()) / 5.0) * 5),
        "deviation": int(round(float(step3.deviation.median()) / 25.0) * 25),
        "backstep": int(round(float(step3.backstep.median()) / 20.0) * 20),
    }
    n = len(d)
    baseline = pd.Series(metrics_for_period(d, 0, n, baseline_cfg["depth"], baseline_cfg["deviation"], baseline_cfg["backstep"], args.point))

    rows = []
    for r in candidates.itertuples(index=False):
        m = metrics_for_period(d, 0, n, r.depth, r.deviation, r.backstep, args.point)
        row = dict(depth=r.depth, deviation=r.deviation, backstep=r.backstep, **m)
        row["deviation_pips"] = r.deviation / 10.0
        row["robustness_score"] = score(pd.Series(row), baseline)
        rows.append(row)

    result = pd.DataFrame(rows)
    result["depth_distance"] = (result.depth - baseline_cfg["depth"]).abs()
    result["deviation_distance"] = (result.deviation - baseline_cfg["deviation"]).abs()
    result["backstep_distance"] = (result.backstep - baseline_cfg["backstep"]).abs()
    result = result.sort_values(["robustness_score", "median_efficiency"], ascending=False)
    result.to_csv(out / "robustness_all_results.csv", index=False)
    result.head(50).to_csv(out / "robustness_top50.csv", index=False)

    for col in ["depth", "deviation", "backstep"]:
        marginal = result.groupby(col, as_index=False).agg(
            configs=("robustness_score", "count"), median_score=("robustness_score", "median"),
            min_score=("robustness_score", "min"), median_efficiency=("median_efficiency", "median"),
            median_swing_pips=("median_swing_pips", "median"), median_bars=("median_bars", "median"),
            median_pivots=("pivot_count", "median"),
        )
        marginal.to_csv(out / f"sensitivity_{col}.csv", index=False)

    plateau = []
    for col in ["depth", "deviation", "backstep"]:
        m = pd.read_csv(out / f"sensitivity_{col}.csv")
        threshold = float(m.median_score.max()) * 0.99
        keep = m[m.median_score >= threshold].copy()
        keep.insert(0, "parameter", col); keep["threshold"] = threshold
        plateau.append(keep)
    pd.concat(plateau, ignore_index=True).to_csv(out / "robust_parameter_plateau.csv", index=False)

    (out / "README.md").write_text(
        "# EURUSD M1 ZigZag Step 4 — Robustness / Sensitivity\n\n"
        f"Candidates are inherited from Step 3: `{args.step3}`.\n"
        f"Baseline center: depth={baseline_cfg['depth']}, deviation={baseline_cfg['deviation']}, backstep={baseline_cfg['backstep']}.\n"
        "Depth is probed every 5 bars across +/-50 bars around the Step 3 center.\n"
        "Deviation preserves the full Step 3 range; backstep preserves Step 3 values and adds the next neighboring value.\n"
        "The robustness score measures retention of efficiency, swing size, swing duration, and pivot density versus the inherited baseline.\n"
        "Plateau output is descriptive and is not a final trading parameter.\n", encoding="utf-8")
    print(f"Candidates: {len(candidates)}")
    print(result.head(20).to_string(index=False))
    print("\nBaseline:", baseline_cfg)


if __name__ == "__main__":
    main()
