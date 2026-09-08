from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data/eurusd/EURUSD_1m.csv"
OUT = ROOT / "data/economic/economic_calendar.csv"
FAILURES = ROOT / "results/step10_economic/calendar_fetch_failures.csv"
FETCH_STATS = ROOT / "results/step10_economic/calendar_fetch_stats.csv"
LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EURUSD-Scalper-AI/1.0; research)"}
MAX_RETRIES = 4
BACKOFF_SECONDS = (5, 15, 30, 60)


def parse_number(value: str):
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text or text.lower() in {"nan", "none", "n/a", "-"}:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m:
        return None
    number = float(m.group(0))
    suffix = text[m.end():].strip().upper()
    if suffix.startswith("K"):
        number *= 1_000
    elif suffix.startswith("M"):
        number *= 1_000_000
    elif suffix.startswith("B"):
        number *= 1_000_000_000
    return number


def higher_is_better(event: str) -> bool:
    name = event.lower()
    lower_better = (
        "unemployment rate", "unemployment change", "jobless claims",
        "initial claims", "continuing claims", "job cuts", "unemployed",
    )
    return not any(term in name for term in lower_better)


def compute_bias(currency: str, event: str, actual: str, forecast: str) -> int:
    a = parse_number(actual)
    f = parse_number(forecast)
    if a is None or f is None or a == f:
        return 0
    good_for_currency = (a > f) if higher_is_better(event) else (a < f)
    currency_sign = 1 if good_for_currency else -1
    return currency_sign if currency == "EUR" else -currency_sign


def price_range() -> tuple[date, date]:
    px = pd.read_csv(PRICE, usecols=["timestamp"])
    ts = pd.to_datetime(px["timestamp"], errors="coerce").dropna()
    if ts.empty:
        raise ValueError("EURUSD price file has no valid timestamps")
    return ts.dt.date.min(), ts.dt.date.max()


def week_key(day: date) -> str:
    return f"{day.strftime('%b').lower()}{day.day}.{day.year}"


def resolve_calendar_date(text: str, week_day: date) -> date | None:
    m = re.search(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})", text)
    if not m:
        return None
    month_num = datetime.strptime(m.group(1), "%b").month
    year = week_day.year
    if week_day.month == 12 and month_num == 1:
        year += 1
    elif week_day.month == 1 and month_num == 12:
        year -= 1
    try:
        return date(year, month_num, int(m.group(2)))
    except ValueError:
        return None


def impact_label(cell) -> str:
    if cell is None:
        return ""

    # Forex Factory has used several HTML representations for impact.
    # Prefer explicit title attributes, then fall back to icon classes/text.
    candidates = []
    for node in cell.find_all(True):
        for attr in ("title", "aria-label", "data-title"):
            value = node.get(attr)
            if value:
                candidates.append(str(value))
        classes = node.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()
        candidates.extend(str(c) for c in classes)

    candidates.append(cell.get_text(" ", strip=True))
    text = " ".join(candidates).upper()

    if "HIGH IMPACT" in text or "IMPACT-RED" in text or "IMPACT_RED" in text:
        return "HIGH"
    if "MEDIUM IMPACT" in text or "MEDIUM" in text or "IMPACT-ORANGE" in text or "IMPACT_ORANGE" in text or "IMPACT-YELLOW" in text or "IMPACT_YELLOW" in text:
        return "MEDIUM"
    if "LOW IMPACT" in text or "IMPACT-GREY" in text or "IMPACT_GRAY" in text or "IMPACT-GRAY" in text:
        return "LOW"
    return ""


def scrape_week(day: date, session: requests.Session) -> list[dict]:
    url = f"https://www.forexfactory.com/calendar?week={week_key(day)}"
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", class_="calendar__table")
            if table is None:
                raise RuntimeError(f"Calendar table not found: {url}")

            rows = []
            current_date = None
            for tr in table.select("tr.calendar__row.calendar_row"):
                def cell(field: str):
                    return tr.select_one(f"td.calendar__cell.calendar__{field}.{field}")

                date_cell = cell("date")
                time_cell = cell("time")
                if date_cell and date_cell.get_text(strip=True):
                    current_date = resolve_calendar_date(date_cell.get_text(" ", strip=True), day)

                currency_cell = cell("currency")
                impact_cell = cell("impact")
                event_cell = cell("event")
                actual_cell = cell("actual")
                forecast_cell = cell("forecast")
                previous_cell = cell("previous")
                if not (current_date and currency_cell and event_cell):
                    continue

                currency = currency_cell.get_text(strip=True).upper()
                if currency not in {"EUR", "USD"}:
                    continue
                impact = impact_label(impact_cell)
                if not impact:
                    continue

                event = event_cell.get_text(" ", strip=True)
                actual = actual_cell.get_text(" ", strip=True) if actual_cell else ""
                forecast = forecast_cell.get_text(" ", strip=True) if forecast_cell else ""
                previous = previous_cell.get_text(" ", strip=True) if previous_cell else ""
                time_text = time_cell.get_text(" ", strip=True) if time_cell else "12:00am"
                if "day" in time_text.lower() or "all day" in time_text.lower():
                    time_text = "12:00am"
                try:
                    local_dt = datetime.strptime(f"{current_date} {time_text}", "%Y-%m-%d %I:%M%p").replace(tzinfo=LONDON)
                except ValueError:
                    continue
                rows.append({
                    "release_time": local_dt.astimezone(UTC).isoformat(),
                    "currency": currency,
                    "impact": impact,
                    "event": event,
                    "bias": compute_bias(currency, event, actual, forecast),
                    "actual": actual,
                    "forecast": forecast,
                    "previous": previous,
                })
            return rows
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_SECONDS[attempt]
                print(f"  attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}; retrying in {delay}s")
                time.sleep(delay)
    raise RuntimeError(last_error or f"Unknown fetch error: {url}")


def main() -> None:
    start, end = price_range()
    cursor = start - timedelta(days=(start.weekday() + 1) % 7)
    rows = []
    failures = []
    requested_weeks = 0

    with requests.Session() as session:
        while cursor <= end:
            requested_weeks += 1
            print(f"Fetching economic calendar week {cursor}")
            try:
                week_rows = scrape_week(cursor, session)
                rows.extend(week_rows)
                print(f"  success: {len(week_rows)} EUR/USD events")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failures.append({
                    "week_start": cursor.isoformat(),
                    "week_key": week_key(cursor),
                    "error": str(exc),
                })
            cursor += timedelta(days=7)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No EUR/USD economic events were downloaded")

    df = df.drop_duplicates(subset=["release_time", "currency", "event"])
    df["release_time"] = pd.to_datetime(df["release_time"], utc=True)
    df = df.sort_values("release_time")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    FAILURES.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = OUT.with_suffix(".tmp")
    df.to_csv(tmp_out, index=False)
    tmp_out.replace(OUT)

    pd.DataFrame(failures, columns=["week_start", "week_key", "error"]).to_csv(FAILURES, index=False)
    pd.DataFrame([{
        "price_start": start.isoformat(),
        "price_end": end.isoformat(),
        "requested_weeks": requested_weeks,
        "successful_weeks": requested_weeks - len(failures),
        "failed_weeks": len(failures),
        "event_count": len(df),
    }]).to_csv(FETCH_STATS, index=False)

    print(f"Saved {len(df)} EUR/USD events to {OUT}")
    print(f"Weeks: {requested_weeks - len(failures)} successful, {len(failures)} failed")
    if failures:
        print(f"WARNING: failed weeks recorded in {FAILURES}")


if __name__ == "__main__":
    main()
