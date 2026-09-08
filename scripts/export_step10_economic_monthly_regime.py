from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/economic/economic_calendar.csv"
SEED = ROOT / "data/economic/monthly_regime_seed.csv"
OUT = ROOT / "results/step10_economic"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_WEIGHTS = {"HIGH": 3.0, "MEDIUM": 1.5, "LOW": 0.5}
FLIP_THRESHOLD = 4.0
CONFIRM_DAYS = 1

DAILY_COLUMNS = [
    "date", "month", "opening_regime", "regime", "previous_regime",
    "regime_changed", "daily_fundamental_score", "cumulative_month_score",
    "high_impact_event_count", "event_count", "confidence",
]
MONTHLY_COLUMNS = [
    "month", "opening_regime", "final_regime", "days", "regime_changes",
    "total_event_count", "high_impact_event_count", "final_cumulative_score",
]


def load_events() -> pd.DataFrame:
    if not EVENTS.exists():
        raise FileNotFoundError(f"Missing {EVENTS}")
    df = pd.read_csv(EVENTS)
    required = {"release_time", "currency", "impact", "bias", "event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Economic calendar missing columns: {sorted(missing)}")
    df["release_time"] = pd.to_datetime(df["release_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["release_time"]).copy()
    df["impact"] = df["impact"].astype(str).str.upper().str.strip()
    df["bias"] = pd.to_numeric(df["bias"], errors="coerce").fillna(0).clip(-1, 1)
    df["weight"] = df["impact"].map(EVENT_WEIGHTS).fillna(0.0)
    df["signed_score"] = df["bias"] * df["weight"]
    df["date"] = df["release_time"].dt.date
    return df.sort_values("release_time").reset_index(drop=True)


def load_seed() -> dict[str, str]:
    if not SEED.exists():
        return {}
    seed = pd.read_csv(SEED)
    if seed.empty or not {"month", "regime"}.issubset(seed.columns):
        return {}
    return dict(zip(seed["month"].astype(str), seed["regime"].astype(str).str.upper()))


def regime_for_score(score: float, previous: str) -> str:
    if score >= FLIP_THRESHOLD:
        return "UP"
    if score <= -FLIP_THRESHOLD:
        return "DOWN"
    return previous


def main() -> None:
    events = load_events()
    seed = load_seed()

    if events.empty:
        daily = pd.DataFrame(columns=DAILY_COLUMNS)
        monthly = pd.DataFrame(columns=MONTHLY_COLUMNS)
        daily.to_csv(OUT / "daily_economic_regime.csv", index=False)
        monthly.to_csv(OUT / "monthly_economic_regime.csv", index=False)
        print("Step 10 — no economic calendar rows supplied; empty outputs created.")
        print(f"Expected input: {EVENTS}")
        return

    rows: list[dict] = []
    current_regime = "NEUTRAL"
    previous_month_final = "NEUTRAL"
    current_month = None
    cumulative_score = 0.0
    days_confirmed = 0
    opening_regime = "NEUTRAL"

    for day in sorted(events["release_time"].dt.tz_convert(None).dt.date.unique()):
        month = pd.Timestamp(day).to_period("M")
        month_key = str(month)

        if current_month != month:
            current_month = month
            current_regime = seed.get(month_key, previous_month_final)
            if current_regime not in {"UP", "DOWN", "NEUTRAL"}:
                current_regime = "NEUTRAL"
            cumulative_score = 0.0
            days_confirmed = 0
            opening_regime = current_regime

        day_events = events[events["date"] == day]
        daily_score = float(day_events["signed_score"].sum())
        cumulative_score += daily_score

        proposed = regime_for_score(cumulative_score, current_regime)
        changed = proposed != current_regime
        if changed:
            days_confirmed += 1
            if days_confirmed >= CONFIRM_DAYS:
                current_regime = proposed
                days_confirmed = 0
                regime_changed = True
            else:
                regime_changed = False
        else:
            days_confirmed = 0
            regime_changed = False

        rows.append({
            "date": day,
            "month": month_key,
            "opening_regime": opening_regime,
            "regime": current_regime,
            "previous_regime": proposed if changed else current_regime,
            "regime_changed": regime_changed,
            "daily_fundamental_score": daily_score,
            "cumulative_month_score": cumulative_score,
            "high_impact_event_count": int((day_events["impact"] == "HIGH").sum()),
            "event_count": int(len(day_events)),
            "confidence": min(1.0, abs(cumulative_score) / FLIP_THRESHOLD),
        })

        previous_month_final = current_regime

    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    daily.to_csv(OUT / "daily_economic_regime.csv", index=False)

    monthly = (
        daily.groupby("month", sort=True)
        .agg(
            opening_regime=("opening_regime", "first"),
            final_regime=("regime", "last"),
            days=("date", "count"),
            regime_changes=("regime_changed", "sum"),
            total_event_count=("event_count", "sum"),
            high_impact_event_count=("high_impact_event_count", "sum"),
            final_cumulative_score=("cumulative_month_score", "last"),
        )
        .reset_index()
    )
    monthly.to_csv(OUT / "monthly_economic_regime.csv", index=False)

    print("Step 10 — Stateful Economic Monthly Regime")
    print(f"Days: {len(daily)} | Months: {len(monthly)}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
