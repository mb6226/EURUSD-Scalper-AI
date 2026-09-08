from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP7 = ROOT / "results/zigzag_step7/swings_with_previous_fibo.csv"
STEP8 = ROOT / "results/zigzag_step8/swings_lwma_trend_aligned.csv"
OUT = ROOT / "results/zigzag_step9"
OUT.mkdir(parents=True, exist_ok=True)

sw7 = pd.read_csv(STEP7)
sw8 = pd.read_csv(STEP8)

aligned = sw8.loc[sw8["swing_trend_aligned"].astype(str).str.lower().eq("true")].copy()
cols = ["swing_id", "direction", "trend", "swing_pips", "bars", "swing_trend_aligned"]
aligned = aligned[cols]

fib_cols = [
    "swing_id", "prior_direction", "prior_swing_pips", "prior_bars",
    "current_end_retracement_pct", "current_end_fib_zone"
]
merged = aligned.merge(sw7[fib_cols], on="swing_id", how="inner", validate="one_to_one")
merged = merged.dropna(subset=["current_end_retracement_pct"]).copy()
merged["reversal_pct"] = merged["current_end_retracement_pct"]

# Explicit unbounded reversal zones: do not cap at 100%.
bins = [-float("inf"), 23.6, 38.2, 50.0, 61.8, 78.6, 100.0, 127.2, 161.8, 200.0, 261.8, 361.8, 423.6, float("inf")]
labels = ["<0", "0-23.6", "23.6-38.2", "38.2-50", "50-61.8", "61.8-78.6", "78.6-100", "100-127.2", "127.2-161.8", "161.8-200", "200-261.8", "261.8-361.8", "361.8-423.6", ">=423.6"]
merged["reversal_fib_zone"] = pd.cut(merged["reversal_pct"], bins=bins, labels=labels, right=False)

summary_rows = []
for group_name, g in [("ALL", merged), ("UP", merged[merged.direction == "UP"]), ("DOWN", merged[merged.direction == "DOWN"])]:
    if g.empty:
        continue
    r = g["reversal_pct"]
    summary_rows.append({
        "group": group_name,
        "count": len(g),
        "median_reversal_pct": r.median(),
        "mean_reversal_pct": r.mean(),
        "p25_reversal_pct": r.quantile(.25),
        "p75_reversal_pct": r.quantile(.75),
        "p90_reversal_pct": r.quantile(.90),
        "min_reversal_pct": r.min(),
        "max_reversal_pct": r.max(),
        "pct_ge_61_8": (r >= 61.8).mean(),
        "pct_ge_78_6": (r >= 78.6).mean(),
        "pct_ge_100": (r >= 100).mean(),
        "pct_ge_127_2": (r >= 127.2).mean(),
        "pct_ge_161_8": (r >= 161.8).mean(),
        "pct_ge_200": (r >= 200).mean(),
        "pct_ge_261_8": (r >= 261.8).mean(),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "aligned_fibo_summary.csv", index=False)

zone = (merged.groupby("reversal_fib_zone", observed=False).size().rename("count").reset_index())
zone["pct"] = zone["count"] / len(merged)
zone.to_csv(OUT / "aligned_fibo_zone_distribution.csv", index=False)

by_direction = (merged.groupby(["direction", "reversal_fib_zone"], observed=False).size().rename("count").reset_index())
by_direction["direction_total"] = by_direction.groupby("direction")["count"].transform("sum")
by_direction["pct_within_direction"] = by_direction["count"] / by_direction["direction_total"]
by_direction.to_csv(OUT / "aligned_fibo_direction_zone.csv", index=False)

by_trend = (merged.groupby(["trend", "reversal_fib_zone"], observed=False).size().rename("count").reset_index())
by_trend["trend_total"] = by_trend.groupby("trend")["count"].transform("sum")
by_trend["pct_within_trend"] = by_trend["count"] / by_trend["trend_total"]
by_trend.to_csv(OUT / "aligned_fibo_trend_zone.csv", index=False)

merged.sort_values("swing_id").to_csv(OUT / "aligned_swings_with_fibo.csv", index=False)

print("Step 9 — aligned swing previous-Fibonacci reversal statistics")
print(f"Aligned swings: {len(aligned)}")
print(f"Fibo-eligible aligned swings: {len(merged)}")
if not summary.empty:
    print(summary.to_string(index=False))
print("\nZone distribution:")
print(zone.to_string(index=False))
