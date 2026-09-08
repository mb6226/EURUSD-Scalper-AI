from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/step15_backtest"
OUT = ROOT / "results/step17_cost_aware"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"A":"A_Breakout","B":"B_Break_Retest","C":"C_Fibonacci_Rejection","D":"D_M1_Structure_Break","E":"E_Momentum_Continuation"}
# Research assumptions: source has no bid/ask history, so costs are explicit assumptions, not broker-observed costs.
BASE_SPREAD_PIPS = 0.8
BASE_SLIPPAGE_PER_SIDE_PIPS = 0.1
SENSITIVITY_ROUNDTRIP_PIPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def load(key):
    t = pd.read_csv(IN / f"step15_trades_{key}.csv")
    req = {"model", "event", "swing_id", "direction", "event_time", "entry_time", "entry_price", "atr14_pips", "exit_time", "exit_price", "exit_reason", "hold_bars", "r_multiple"}
    missing = req - set(t.columns)
    if missing:
        raise ValueError(f"Step 15 trade file missing columns: {sorted(missing)}")
    for c in ["atr14_pips", "hold_bars", "r_multiple", "entry_price", "exit_price"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    for c in ["event_time", "entry_time", "exit_time"]:
        t[c] = pd.to_datetime(t[c], errors="coerce")
    return t.dropna(subset=["atr14_pips", "r_multiple", "entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)


def apply_costs(t, roundtrip_pips):
    x = t.copy()
    x["cost_pips"] = float(roundtrip_pips)
    x["cost_R"] = x["cost_pips"] / x["atr14_pips"]
    x["gross_R"] = x["r_multiple"]
    x["net_R"] = x["gross_R"] - x["cost_R"]
    return x


def metrics(t):
    if t.empty:
        return {"trades": 0}
    r = t["net_R"]
    gross = t["gross_R"]
    wins = r > 0
    losses = r < 0
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    eq = r.cumsum()
    dd = eq - eq.cummax()
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    start = t["entry_time"].min()
    end = t["exit_time"].max()
    years = max((end - start).total_seconds() / (365.25 * 86400), 1 / 365.25)
    return {
        "trades": int(len(r)),
        "wins_net": int(wins.sum()),
        "losses_net": int(losses.sum()),
        "win_rate_net_pct": 100 * wins.mean(),
        "gross_total_R": gross.sum(),
        "cost_total_R": t["cost_R"].sum(),
        "net_total_R": r.sum(),
        "gross_expectancy_R": gross.mean(),
        "net_expectancy_R": r.mean(),
        "net_median_R": r.median(),
        "net_std_R": r.std(ddof=1) if len(r) > 1 else 0.0,
        "net_profit_factor": gp / gl if gl > 0 else np.inf,
        "net_max_drawdown_R": dd.min(),
        "trade_sharpe": sharpe,
        "trades_per_year": len(r) / years,
        "avg_hold_bars": t["hold_bars"].mean(),
        "median_hold_bars": t["hold_bars"].median(),
        "start_time": start,
        "end_time": end,
        "avg_cost_pips": t["cost_pips"].mean(),
        "avg_cost_R": t["cost_R"].mean(),
    }


def main():
    baseline_roundtrip = BASE_SPREAD_PIPS + 2 * BASE_SLIPPAGE_PER_SIDE_PIPS
    summary = []
    sensitivity = []
    all_baseline = []

    for key, name in MODELS.items():
        gross = load(key)
        baseline = apply_costs(gross, baseline_roundtrip)
        all_baseline.append(baseline.assign(model=name))
        m = metrics(baseline)
        summary.append({"model": name, "cost_scenario": "BASELINE", "roundtrip_cost_pips": baseline_roundtrip, **m})
        for cost in SENSITIVITY_ROUNDTRIP_PIPS:
            x = apply_costs(gross, cost)
            mm = metrics(x)
            sensitivity.append({"model": name, "roundtrip_cost_pips": cost, **mm})

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "step17_model_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "step17_cost_sensitivity.csv", index=False)

    trades = pd.concat(all_baseline, ignore_index=True).sort_values(["model", "entry_time"])
    trades["equity_R"] = trades.groupby("model")["net_R"].cumsum()
    trades["drawdown_R"] = trades.groupby("model")["equity_R"].transform(lambda x: x - x.cummax())
    trades.to_csv(OUT / "step17_baseline_trades.csv", index=False)
    trades[["model", "swing_id", "direction", "entry_time", "exit_time", "gross_R", "cost_pips", "cost_R", "net_R", "equity_R", "drawdown_R"]].to_csv(OUT / "step17_equity_curve.csv", index=False)

    config = {
        "purpose": "Step 17 - Cost-aware backtest",
        "source": "Step 15 trade-level CSVs",
        "cost_model": "Fixed round-trip cost in pips = spread + 2 * per-side slippage; deducted from each trade using trade-specific ATR14 risk.",
        "baseline_spread_pips": BASE_SPREAD_PIPS,
        "baseline_slippage_per_side_pips": BASE_SLIPPAGE_PER_SIDE_PIPS,
        "baseline_roundtrip_cost_pips": baseline_roundtrip,
        "sensitivity_roundtrip_cost_pips": SENSITIVITY_ROUNDTRIP_PIPS,
        "price_data_limitation": "Historical OHLC has no bid/ask spread or observed execution slippage; costs are explicit research assumptions.",
        "entry_rule": "Inherited from Step 15: event at bar close, entry at NEXT_BAR_OPEN.",
        "sl_tp": "Inherited from Step 15: SL 1.0 x ATR14, TP 1.5 x ATR14, fixed at entry.",
        "optimization": False,
        "sharpe_definition": "Trade-level Sharpe = mean(net R) / std(net R) * sqrt(number of trades); not calendar-time annualized.",
        "turnover_proxy": "Trades per year; notional turnover is not available because position sizing/notional is not specified.",
    }
    (OUT / "step17_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print("Step 17 - Cost-aware backtest complete")
    print(s[["model", "roundtrip_cost_pips", "trades", "win_rate_net_pct", "net_expectancy_R", "net_profit_factor", "net_total_R", "net_max_drawdown_R", "trade_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
