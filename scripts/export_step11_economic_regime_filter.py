from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP9 = ROOT / "results/zigzag_step9/aligned_swings_with_fibo.csv"
STEP8 = ROOT / "results/zigzag_step8/swings_with_lwma_trend.csv"
DAILY = ROOT / "results/step10_economic/daily_economic_regime.csv"
OUT = ROOT / "results/step11_economic_regime"
OUT.mkdir(parents=True, exist_ok=True)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Step 11 input: {path}")


def main() -> None:
    for p in (STEP9, STEP8, DAILY):
        require(p)

    step9 = pd.read_csv(STEP9)
    step8 = pd.read_csv(STEP8, usecols=["swing_id", "start_time", "end_time"])
    daily = pd.read_csv(DAILY)

    required9 = {"swing_id", "direction", "trend", "swing_pips", "bars", "reversal_pct", "reversal_fib_zone"}
    required8 = {"swing_id", "start_time", "end_time"}
    require_daily = {"date", "month", "regime", "opening_regime", "previous_regime", "regime_changed", "cumulative_month_score", "confidence"}
    if not required9.issubset(step9.columns):
        raise ValueError(f"Step 9 missing columns: {sorted(required9 - set(step9.columns))}")
    if not required8.issubset(step8.columns):
        raise ValueError(f"Step 8 missing columns: {sorted(required8 - set(step8.columns))}")
    if not require_daily.issubset(daily.columns):
        raise ValueError(f"Step 10 daily data missing columns: {sorted(require_daily - set(daily.columns))}")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date
    if daily["date"].isna().any():
        raise ValueError("Step 10 daily regime contains invalid dates")
    if daily["date"].duplicated().any():
        raise ValueError("Step 10 daily regime has duplicate dates")
    if len(daily) != 311:
        raise ValueError(f"Expected 311 Step 10 trading days, found {len(daily)}")
    bad_regimes = set(daily["regime"].astype(str)) - {"UP", "DOWN", "NEUTRAL"}
    if bad_regimes:
        raise ValueError(f"Unexpected Step 10 regimes: {sorted(bad_regimes)}")

    step8["end_time"] = pd.to_datetime(step8["end_time"], errors="coerce")
    if step8["end_time"].isna().any():
        raise ValueError("Step 8 contains invalid end_time values")
    if step8["swing_id"].duplicated().any():
        raise ValueError("Step 8 has duplicate swing_id values")

    # Step 9 contains the 310 Fibo-eligible aligned swings. Use the swing endpoint
    # date to join the already-built point-in-time daily economic regime. This is
    # explicitly a research filter; no future event information is introduced here.
    merged = step9.merge(step8, on="swing_id", how="left", validate="one_to_one")
    if merged["end_time"].isna().any():
        missing = int(merged["end_time"].isna().sum())
        raise ValueError(f"Could not recover end_time for {missing} Step 9 swings")
    merged["signal_date"] = merged["end_time"].dt.date
    merged = merged.merge(
        daily[["date", "month", "regime", "opening_regime", "previous_regime", "regime_changed", "cumulative_month_score", "confidence"]],
        left_on="signal_date", right_on="date", how="left", validate="many_to_one", suffixes=("", "_daily"),
    )
    if merged["regime"].isna().any():
        missing_dates = merged.loc[merged["regime"].isna(), "signal_date"].astype(str).unique().tolist()
        raise ValueError(f"Step 9 signals outside Step 10 daily coverage: {missing_dates[:10]}")

    merged["economic_regime"] = merged["regime"]
    merged["economic_alignment"] = merged["direction"].eq(merged["economic_regime"])
    merged["economic_filter_status"] = merged["economic_regime"].map({"UP": "KEEP_IF_UP", "DOWN": "KEEP_IF_DOWN", "NEUTRAL": "EXCLUDE_NEUTRAL"})
    merged["economic_regime_match"] = merged["economic_alignment"]
    merged["passes_step11_filter"] = merged["economic_regime"].isin({"UP", "DOWN"}) & merged["economic_alignment"]

    filtered = merged.loc[merged["passes_step11_filter"]].copy().sort_values("swing_id")
    merged = merged.sort_values("swing_id")
    merged.to_csv(OUT / "step11_aligned_signals_with_economic_regime.csv", index=False)
    filtered.to_csv(OUT / "step11_filtered_signals.csv", index=False)

    # Daily regime distribution: this answers the 311-day question directly.
    counts = daily["regime"].value_counts().reindex(["UP", "DOWN", "NEUTRAL"], fill_value=0)
    daily_counts = pd.DataFrame({
        "regime": counts.index,
        "days": counts.values,
        "pct_of_311_days": counts.values / len(daily) * 100.0,
    })
    daily_counts.to_csv(OUT / "step11_daily_regime_counts.csv", index=False)

    # Signal-level filter summary.
    summary = pd.DataFrame([
        {"metric": "step10_trading_days", "value": len(daily)},
        {"metric": "step10_up_days", "value": int(counts["UP"])},
        {"metric": "step10_down_days", "value": int(counts["DOWN"])},
        {"metric": "step10_neutral_days", "value": int(counts["NEUTRAL"])},
        {"metric": "step9_aligned_fibo_signals", "value": len(step9)},
        {"metric": "signals_with_step10_regime", "value": len(merged)},
        {"metric": "economic_aligned_signals", "value": int(merged["economic_alignment"].sum())},
        {"metric": "economic_misaligned_signals", "value": int((~merged["economic_alignment"]).sum())},
        {"metric": "neutral_regime_signals", "value": int((merged["economic_regime"] == "NEUTRAL").sum())},
        {"metric": "step11_kept_signals", "value": len(filtered)},
        {"metric": "step11_kept_pct_of_step9", "value": len(filtered) / len(step9) * 100.0 if len(step9) else 0.0},
    ])
    summary.to_csv(OUT / "step11_filter_summary.csv", index=False)

    # Breakdown by signal direction and economic regime.
    cross = merged.groupby(["direction", "economic_regime"], dropna=False).size().rename("count").reset_index()
    cross["pct_of_direction"] = cross["count"] / cross.groupby("direction")["count"].transform("sum") * 100.0
    cross["passes_filter"] = cross["direction"].eq(cross["economic_regime"]) & cross["economic_regime"].isin({"UP", "DOWN"})
    cross.to_csv(OUT / "step11_direction_regime_matrix.csv", index=False)

    # Monthly signal-level report, preserving PIT daily regime at each endpoint.
    monthly = merged.groupby("month", as_index=False).agg(
        signals=("swing_id", "count"),
        kept=("passes_step11_filter", "sum"),
        up_signals=("direction", lambda x: int((x == "UP").sum())),
        down_signals=("direction", lambda x: int((x == "DOWN").sum())),
        up_regime_signals=("economic_regime", lambda x: int((x == "UP").sum())),
        down_regime_signals=("economic_regime", lambda x: int((x == "DOWN").sum())),
        neutral_regime_signals=("economic_regime", lambda x: int((x == "NEUTRAL").sum())),
    )
    monthly["kept_pct"] = monthly["kept"] / monthly["signals"] * 100.0
    monthly.to_csv(OUT / "step11_monthly_filter_stats.csv", index=False)

    print("Step 11 — Economic Regime Filter")
    print(f"Step 10 days: {len(daily)} | UP={counts['UP']} | DOWN={counts['DOWN']} | NEUTRAL={counts['NEUTRAL']}")
    print(f"Step 9 aligned Fibo signals: {len(step9)}")
    print(f"Kept after economic regime alignment: {len(filtered)} ({len(filtered) / len(step9) * 100.0:.2f}%)")
    print("\nDaily regime counts:")
    print(daily_counts.to_string(index=False))
    print("\nDirection × economic regime:")
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
