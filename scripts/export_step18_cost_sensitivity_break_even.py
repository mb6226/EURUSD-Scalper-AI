from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/step17_cost_aware/step17_cost_sensitivity.csv"
OUT = ROOT / "results/step18_cost_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_COSTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def load() -> pd.DataFrame:
    if not IN.exists():
        raise FileNotFoundError(f"Missing Step 17 sensitivity file: {IN}")
    df = pd.read_csv(IN)
    required = {
        "model", "roundtrip_cost_pips", "trades", "gross_expectancy_R",
        "net_expectancy_R", "net_total_R", "net_profit_factor",
        "net_max_drawdown_R", "trade_sharpe", "avg_cost_R",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Step 17 sensitivity file missing columns: {sorted(missing)}")
    for c in [
        "roundtrip_cost_pips", "gross_expectancy_R", "net_expectancy_R",
        "net_total_R", "net_profit_factor", "net_max_drawdown_R",
        "trade_sharpe", "avg_cost_R",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    return df.dropna(subset=["roundtrip_cost_pips", "gross_expectancy_R", "net_expectancy_R", "avg_cost_R"])


def break_even_from_sensitivity(g: pd.DataFrame) -> tuple[float, str]:
    """Estimate zero-expectancy cost using the observed sensitivity grid.

    If the curve crosses zero between adjacent observed costs, linearly interpolate.
    If it does not cross, fall back to the exact linear cost model implied by
    net_R = gross_R - cost_pips / ATR14, using gross expectancy and avg cost_R at 1 pip.
    """
    g = g.sort_values("roundtrip_cost_pips").reset_index(drop=True)
    x = g["roundtrip_cost_pips"].to_numpy(float)
    y = g["net_expectancy_R"].to_numpy(float)

    for i in range(len(g) - 1):
        if y[i] == 0:
            return float(x[i]), "observed_grid"
        if y[i] * y[i + 1] < 0:
            return float(x[i] + (0.0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])), "linear_interpolation"

    gross = float(g["gross_expectancy_R"].iloc[0])
    one_pip = g[np.isclose(g["roundtrip_cost_pips"], 1.0)]
    if not one_pip.empty and float(one_pip["avg_cost_R"].iloc[0]) > 0:
        cost_R_per_pip = float(one_pip["avg_cost_R"].iloc[0])
    else:
        positive = g[g["roundtrip_cost_pips"] > 0]
        cost_R_per_pip = float((positive["avg_cost_R"] / positive["roundtrip_cost_pips"]).median())
    if cost_R_per_pip <= 0:
        return float("nan"), "unavailable"
    return float(gross / cost_R_per_pip), "analytical_cost_model"


def main() -> None:
    df = load()
    models = sorted(df["model"].unique())
    expected_models = 5
    if len(models) != expected_models:
        raise ValueError(f"Expected 5 Step 15 entry models, found {len(models)}: {models}")

    rows = []
    sensitivity_rows = []
    for model in models:
        g = df[df["model"] == model].copy().sort_values("roundtrip_cost_pips")
        observed_costs = sorted(g["roundtrip_cost_pips"].round(10).unique().tolist())
        if observed_costs != EXPECTED_COSTS:
            raise ValueError(f"Unexpected cost grid for {model}: {observed_costs}")

        be_cost, method = break_even_from_sensitivity(g)
        zero = g[np.isclose(g["roundtrip_cost_pips"], 0.0)].iloc[0]
        half = g[np.isclose(g["roundtrip_cost_pips"], 0.5)].iloc[0]
        one = g[np.isclose(g["roundtrip_cost_pips"], 1.0)].iloc[0]
        gross = float(zero["gross_expectancy_R"])
        cost_per_pip_R = float(one["avg_cost_R"])

        positive_grid = g[g["net_expectancy_R"] > 0]
        max_positive_grid_cost = float(positive_grid["roundtrip_cost_pips"].max()) if not positive_grid.empty else np.nan
        first_negative = g[g["net_expectancy_R"] < 0]
        first_negative_cost = float(first_negative["roundtrip_cost_pips"].min()) if not first_negative.empty else np.nan

        rows.append({
            "model": model,
            "trades": int(g["trades"].iloc[0]),
            "gross_expectancy_R": gross,
            "break_even_roundtrip_cost_pips": be_cost,
            "break_even_method": method,
            "avg_cost_R_per_1pip": cost_per_pip_R,
            "net_expectancy_at_0pip_R": float(zero["net_expectancy_R"]),
            "net_expectancy_at_0_5pip_R": float(half["net_expectancy_R"]),
            "net_expectancy_at_1pip_R": float(one["net_expectancy_R"]),
            "max_positive_grid_cost_pips": max_positive_grid_cost,
            "first_negative_grid_cost_pips": first_negative_cost,
            "cost_efficiency_rank": np.nan,
        })

        sensitivity_rows.extend(g.to_dict("records"))

    summary = pd.DataFrame(rows)
    summary["cost_efficiency_rank"] = summary["break_even_roundtrip_cost_pips"].rank(method="min", ascending=False).astype(int)
    summary = summary.sort_values(["cost_efficiency_rank", "model"]).reset_index(drop=True)

    summary.to_csv(OUT / "step18_break_even_summary.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "step18_cost_sensitivity.csv", index=False)

    # Compact comparison at the three most informative points.
    comparison = summary[[
        "model", "gross_expectancy_R", "break_even_roundtrip_cost_pips",
        "net_expectancy_at_0pip_R", "net_expectancy_at_0_5pip_R",
        "net_expectancy_at_1pip_R", "cost_efficiency_rank",
    ]].copy()
    comparison.to_csv(OUT / "step18_model_comparison.csv", index=False)

    config = {
        "purpose": "Step 18 - Cost sensitivity and break-even analysis",
        "source": str(IN.relative_to(ROOT)),
        "cost_grid_roundtrip_pips": EXPECTED_COSTS,
        "break_even_definition": "Round-trip cost where net expectancy reaches 0R.",
        "break_even_method": "Use linear interpolation when the observed sensitivity grid brackets zero; otherwise use the linear cost model implied by gross expectancy and average 1-pip cost in R.",
        "cost_model_inherited_from_step17": "net_R = gross_R - roundtrip_cost_pips / trade_ATR14_pips",
        "ranking": "Higher break-even round-trip cost means greater cost tolerance.",
        "not_a_profitability_claim": "Break-even is conditional on Step 17's entry/SL/TP rules and explicit cost assumptions; it is not evidence of live profitability or broker-executable edge.",
    }
    (OUT / "step18_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Step 18 - Cost sensitivity + break-even complete")
    print(summary[[
        "model", "gross_expectancy_R", "break_even_roundtrip_cost_pips",
        "net_expectancy_at_0_5pip_R", "net_expectancy_at_1pip_R",
        "cost_efficiency_rank",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
