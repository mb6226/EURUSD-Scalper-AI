from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/economic/economic_calendar.csv"
PRICE = ROOT / "data/eurusd/EURUSD_1m.csv"
SEED = ROOT / "data/economic/monthly_regime_seed.csv"
OUT = ROOT / "results/step10_economic"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_WEIGHTS = {"HIGH": 3.0, "MEDIUM": 1.5, "LOW": 0.5}
FLIP_THRESHOLD = 4.0

DAILY_COLUMNS = [
    "date", "month", "opening_regime", "regime", "previous_regime",
    "regime_changed", "daily_fundamental_score", "cumulative_month_score",
    "high_impact_event_count", "event_count", "confidence",
]
EVENT_STATE_COLUMNS = [
    "release_time", "date", "currency", "impact", "event", "bias",
    "event_score", "regime_before", "regime_after", "cumulative_month_score",
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
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df = df[df["currency"].isin({"EUR", "USD"})].copy()
    df["impact"] = df["impact"].astype(str).str.upper().str.strip()
    df["bias"] = pd.to_numeric(df["bias"], errors="coerce").fillna(0).clip(-1, 1)
    df["weight"] = df["impact"].map(EVENT_WEIGHTS).fillna(0.0)
    df["event_score"] = df["bias"] * df["weight"]
    df["date"] = df["release_time"].dt.tz_convert(None).dt.date
    return df.sort_values("release_time").reset_index(drop=True)


def load_trading_days() -> list:
    if not PRICE.exists():
        raise FileNotFoundError(f"Missing {PRICE}")
    px = pd.read_csv(PRICE, usecols=["timestamp"])
    ts = pd.to_datetime(px["timestamp"], errors="coerce").dropna()
    return sorted(ts.dt.date.unique())


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


def confidence(score: float, regime: str) -> float:
    if regime == "UP":
        return min(1.0, max(0.0, score / FLIP_THRESHOLD))
    if regime == "DOWN":
        return min(1.0, max(0.0, -score / FLIP_THRESHOLD))
    return min(1.0, abs(score) / FLIP_THRESHOLD)


def main() -> None:
    events = load_events()
    trading_days = load_trading_days()
    seed = load_seed()

    if not trading_days:
        raise ValueError("EURUSD price file contains no valid timestamps")

    event_groups = {day: g for day, g in events.groupby("date", sort=False)}
    current_regime = "NEUTRAL"
    previous_month_final = "NEUTRAL"
    current_month = None
    opening_regime = "NEUTRAL"
    cumulative_score = 0.0

    daily_rows: list[dict] = []
    event_rows: list[dict] = []
    monthly_rows: list[dict] = []
    month_start = None
    month_event_count = 0
    month_high_count = 0
    month_changes = 0

    for day in trading_days:
        month = pd.Timestamp(day).to_period("M")
        month_key = str(month)

        if current_month != month:
            if current_month is not None:
                monthly_rows.append({
                    "month": str(current_month),
                    "opening_regime": opening_regime,
                    "final_regime": previous_month_final,
                    "days": sum(pd.Timestamp(d).to_period("M") == current_month for d in trading_days if month_start <= d < day),
                    "regime_changes": month_changes,
                    "total_event_count": month_event_count,
                    "high_impact_event_count": month_high_count,
                    "final_cumulative_score": cumulative_score,
                })

            current_month = month
            current_regime = seed.get(month_key, previous_month_final)
            if current_regime not in {"UP", "DOWN", "NEUTRAL"}:
                current_regime = "NEUTRAL"
            opening_regime = current_regime
            cumulative_score = 0.0
            month_start = day
            month_event_count = 0
            month_high_count = 0
            month_changes = 0

        day_events = event_groups.get(day)
        daily_score = 0.0
        day_high = 0
        day_count = 0
        regime_changed = False

        if day_events is not None:
            for _, ev in day_events.iterrows():
                before = current_regime
                score = float(ev["event_score"])
                cumulative_score += score
                daily_score += score
                day_count += 1
                if ev["impact"] == "HIGH":
                    day_high += 1
                proposed = regime_for_score(cumulative_score, current_regime)
                if proposed != current_regime:
                    current_regime = proposed
                    regime_changed = True
                    month_changes += 1
                event_rows.append({
                    "release_time": ev["release_time"].isoformat(),
                    "date": day,
                    "currency": ev["currency"],
                    "impact": ev["impact"],
                    "event": ev["event"],
                    "bias": float(ev["bias"]),
                    "event_score": score,
                    "regime_before": before,
                    "regime_after": current_regime,
                    "cumulative_month_score": cumulative_score,
                })

        month_event_count += day_count
        month_high_count += day_high
        daily_rows.append({
            "date": day,
            "month": month_key,
            "opening_regime": opening_regime,
            "regime": current_regime,
            "previous_regime": current_regime if not regime_changed else ("DOWN" if current_regime == "UP" else "UP"),
            "regime_changed": regime_changed,
            "daily_fundamental_score": daily_score,
            "cumulative_month_score": cumulative_score,
            "high_impact_event_count": day_high,
            "event_count": day_count,
            "confidence": confidence(cumulative_score, current_regime),
        })
        previous_month_final = current_regime

    if current_month is not None:
        monthly_rows.append({
            "month": str(current_month),
            "opening_regime": opening_regime,
            "final_regime": previous_month_final,
            "days": sum(pd.Timestamp(d).to_period("M") == current_month for d in trading_days if month_start <= d <= trading_days[-1]),
            "regime_changes": month_changes,
            "total_event_count": month_event_count,
            "high_impact_event_count": month_high_count,
            "final_cumulative_score": cumulative_score,
        })

    daily = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS)
    event_state = pd.DataFrame(event_rows, columns=EVENT_STATE_COLUMNS)
    monthly = pd.DataFrame(monthly_rows, columns=MONTHLY_COLUMNS)

    daily.to_csv(OUT / "daily_economic_regime.csv", index=False)
    event_state.to_csv(OUT / "economic_event_state.csv", index=False)
    monthly.to_csv(OUT / "monthly_economic_regime.csv", index=False)

    print("Step 10 — Stateful Economic Monthly Regime")
    print(f"Trading days: {len(daily)} | Months: {len(monthly)} | Events: {len(event_state)}")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
