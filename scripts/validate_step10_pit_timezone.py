from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = ROOT / "data/economic/economic_calendar.csv"
EVENT_STATE = ROOT / "results/step10_economic/economic_event_state.csv"
DAILY = ROOT / "results/step10_economic/economic_daily_regime.csv"
OUT = ROOT / "results/step10_economic/pit_timezone_validation.csv"

UTC = ZoneInfo("UTC")
LONDON = ZoneInfo("Europe/London")
BERLIN = ZoneInfo("Europe/Berlin")


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def main() -> None:
    cal = pd.read_csv(CALENDAR)
    required = {"release_time", "currency", "impact", "event", "bias"}
    missing = required - set(cal.columns)
    if missing:
        raise RuntimeError(f"Calendar missing columns: {sorted(missing)}")

    cal["release_time"] = parse_utc(cal["release_time"])
    if cal["release_time"].isna().any():
        raise RuntimeError("Calendar contains invalid release_time values")
    if not cal["release_time"].is_monotonic_increasing:
        raise RuntimeError("Calendar release_time is not sorted")

    duplicate_keys = cal.duplicated(["release_time", "currency", "event"]).sum()
    if duplicate_keys:
        raise RuntimeError(f"Calendar contains {duplicate_keys} duplicate release keys")

    checks = []
    checks.append(("calendar_rows", len(cal), "PASS"))
    checks.append(("calendar_invalid_timestamps", int(cal["release_time"].isna().sum()), "PASS"))
    checks.append(("calendar_duplicate_release_keys", int(duplicate_keys), "PASS"))

    # DST sanity: UTC conversion must differ by one hour between winter/summer
    # for the same nominal London local time. This catches fixed-offset mistakes.
    winter = datetime(2025, 1, 15, 9, 30, tzinfo=LONDON).astimezone(UTC).hour
    summer = datetime(2025, 7, 15, 9, 30, tzinfo=LONDON).astimezone(UTC).hour
    checks.append(("london_dst_winter_0930_utc_hour", winter, "PASS" if winter == 9 else "FAIL"))
    checks.append(("london_dst_summer_0930_utc_hour", summer, "PASS" if summer == 8 else "FAIL"))

    # The stored UTC timestamp must round-trip without changing the instant.
    roundtrip = cal["release_time"].dt.tz_convert(LONDON).dt.tz_convert(UTC)
    roundtrip_ok = bool((roundtrip == cal["release_time"]).all())
    checks.append(("utc_roundtrip_integrity", int(roundtrip_ok), "PASS" if roundtrip_ok else "FAIL"))

    # PIT event-state validation: event state must be chronologically ordered and
    # regime_before must represent the state immediately before that event.
    if EVENT_STATE.exists():
        ev = pd.read_csv(EVENT_STATE)
        if not ev.empty:
            ev["release_time"] = parse_utc(ev["release_time"])
            ordered = bool(ev["release_time"].is_monotonic_increasing)
            checks.append(("event_state_sorted_by_release_time", int(ordered), "PASS" if ordered else "FAIL"))
            required_ev = {"release_time", "regime_before", "regime_after", "cumulative_month_score"}
            checks.append(("event_state_required_columns", int(required_ev <= set(ev.columns)), "PASS" if required_ev <= set(ev.columns) else "FAIL"))
        else:
            checks.append(("event_state_nonempty", 0, "FAIL"))
    else:
        checks.append(("event_state_exists", 0, "FAIL"))

    # Daily regime must have one opening regime per month and remain populated.
    if DAILY.exists():
        daily = pd.read_csv(DAILY)
        required_daily = {"date", "month", "opening_regime", "regime", "regime_changed"}
        ok_cols = required_daily <= set(daily.columns)
        checks.append(("daily_required_columns", int(ok_cols), "PASS" if ok_cols else "FAIL"))
        if ok_cols and not daily.empty:
            openings = daily.groupby("month")["opening_regime"].nunique()
            stable_openings = bool((openings <= 1).all())
            checks.append(("one_opening_regime_per_month", int(stable_openings), "PASS" if stable_openings else "FAIL"))
            changes = daily.loc[daily["regime_changed"].astype(bool)]
            checks.append(("daily_regime_change_rows", len(changes), "PASS"))
    else:
        checks.append(("daily_regime_exists", 0, "FAIL"))

    result = pd.DataFrame(checks, columns=["check", "value", "status"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    failures = result[result["status"] == "FAIL"]
    print(result.to_string(index=False))
    if not failures.empty:
        raise RuntimeError(f"Step 10 PIT/timezone validation failed: {len(failures)} checks")
    print(f"Step 10 PIT/timezone validation PASSED: {len(result)} checks")


if __name__ == "__main__":
    main()
