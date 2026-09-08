from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = ROOT / "data/economic/economic_calendar.csv"
FAILURES = ROOT / "results/step10_economic/calendar_fetch_failures.csv"
FETCH_STATS = ROOT / "results/step10_economic/calendar_fetch_stats.csv"
EVENT_STATE = ROOT / "results/step10_economic/economic_event_state.csv"
DAILY = ROOT / "results/step10_economic/daily_economic_regime.csv"
OUT = ROOT / "results/step10_economic/pit_timezone_validation.csv"
UTC = ZoneInfo("UTC")
LONDON = ZoneInfo("Europe/London")


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def main() -> None:
    checks = []
    if FETCH_STATS.exists():
        stats = pd.read_csv(FETCH_STATS)
        failed = int(stats.iloc[0].get("failed_weeks", -1)) if not stats.empty else -1
        requested = int(stats.iloc[0].get("requested_weeks", -1)) if not stats.empty else -1
        successful = int(stats.iloc[0].get("successful_weeks", -1)) if not stats.empty else -1
        checks.append(("historical_calendar_fetch_complete", int(failed == 0 and requested > 0 and successful == requested), "PASS" if failed == 0 and requested > 0 and successful == requested else "FAIL"))
    else:
        checks.append(("fetch_stats_exists", 0, "FAIL"))
    if FAILURES.exists():
        failures = pd.read_csv(FAILURES)
        checks.append(("calendar_fetch_failure_rows", len(failures), "PASS" if failures.empty else "FAIL"))
    else:
        checks.append(("calendar_fetch_failures_file_exists", 0, "FAIL"))

    cal = pd.read_csv(CALENDAR)
    required = {"release_time", "currency", "impact", "event", "bias"}
    missing = required - set(cal.columns)
    if missing: raise RuntimeError(f"Calendar missing columns: {sorted(missing)}")
    cal["release_time"] = parse_utc(cal["release_time"])
    invalid = int(cal["release_time"].isna().sum())
    duplicate_keys = int(cal.duplicated(["release_time", "currency", "event"]).sum())
    ordered = bool(cal["release_time"].is_monotonic_increasing)
    checks += [("calendar_rows", len(cal), "PASS"), ("calendar_invalid_timestamps", invalid, "PASS" if invalid == 0 else "FAIL"), ("calendar_duplicate_release_keys", duplicate_keys, "PASS" if duplicate_keys == 0 else "FAIL"), ("calendar_release_time_sorted", int(ordered), "PASS" if ordered else "FAIL")]

    winter = datetime(2025, 1, 15, 9, 30, tzinfo=LONDON).astimezone(UTC).hour
    summer = datetime(2025, 7, 15, 9, 30, tzinfo=LONDON).astimezone(UTC).hour
    checks += [("london_dst_winter_0930_utc_hour", winter, "PASS" if winter == 9 else "FAIL"), ("london_dst_summer_0930_utc_hour", summer, "PASS" if summer == 8 else "FAIL")]
    roundtrip = cal["release_time"].dt.tz_convert(LONDON).dt.tz_convert(UTC)
    roundtrip_ok = bool((roundtrip == cal["release_time"]).all())
    checks.append(("utc_roundtrip_integrity", int(roundtrip_ok), "PASS" if roundtrip_ok else "FAIL"))

    if EVENT_STATE.exists():
        ev = pd.read_csv(EVENT_STATE)
        if not ev.empty:
            ev["release_time"] = parse_utc(ev["release_time"])
            ordered = bool(ev["release_time"].is_monotonic_increasing)
            required_ev = {"release_time", "regime_before", "regime_after", "cumulative_month_score"}
            checks += [("event_state_sorted_by_release_time", int(ordered), "PASS" if ordered else "FAIL"), ("event_state_required_columns", int(required_ev <= set(ev.columns)), "PASS" if required_ev <= set(ev.columns) else "FAIL")]
        else: checks.append(("event_state_nonempty", 0, "FAIL"))
    else: checks.append(("event_state_exists", 0, "FAIL"))

    if DAILY.exists():
        daily = pd.read_csv(DAILY)
        required_daily = {"date", "month", "opening_regime", "regime", "regime_changed"}
        ok_cols = required_daily <= set(daily.columns)
        checks.append(("daily_required_columns", int(ok_cols), "PASS" if ok_cols else "FAIL"))
        if ok_cols and not daily.empty:
            openings = daily.groupby("month")["opening_regime"].nunique()
            stable = bool((openings <= 1).all())
            checks.append(("one_opening_regime_per_month", int(stable), "PASS" if stable else "FAIL"))
    else: checks.append(("daily_regime_exists", 0, "FAIL"))

    result = pd.DataFrame(checks, columns=["check", "value", "status"])
    OUT.parent.mkdir(parents=True, exist_ok=True); result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    failures = result[result["status"] == "FAIL"]
    if not failures.empty: raise RuntimeError(f"Step 10 PIT/timezone validation failed: {len(failures)} checks")
    print(f"Step 10 PIT/timezone validation PASSED: {len(result)} checks")


if __name__ == "__main__": main()
