from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
UTC = ZoneInfo("UTC")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EURUSD-Scalper-AI/1.0; research)"}
MAX_RETRIES = 4
BACKOFF_SECONDS = (5, 15, 30, 60)
TIME_RE = re.compile(r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)$", re.IGNORECASE)
DATE_RE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?$", re.IGNORECASE)


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
    normalized = " ".join(str(text).split())
    m = DATE_RE.fullmatch(normalized)
    if not m:
        return None
    month_num = datetime.strptime(m.group("month"), "%b").month
    year = week_day.year
    if week_day.month == 12 and month_num == 1:
        year += 1
    elif week_day.month == 1 and month_num == 12:
        year -= 1
    try:
        return date(year, month_num, int(m.group("day")))
    except ValueError:
        return None


def _class_tokens(tag) -> set[str]:
    if tag is None:
        return set()
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    return {str(x) for x in classes}


def extract_page_timezone(soup: BeautifulSoup) -> tuple[str, ZoneInfo]:
    """Extract the complete advertised IANA timezone without accepting fragments."""
    text = soup.get_text(" ", strip=True)
    candidates = []
    for pattern in (
        r"Calendar\s+Time\s+Zone\s*:\s*([A-Za-z_]+/[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+)*)",
        r"Time\s+Zone\s*:\s*([A-Za-z_]+/[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+)*)",
    ):
        candidates.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    for name in candidates:
        try:
            return name, ZoneInfo(name)
        except ZoneInfoNotFoundError:
            continue

    label_map = {
        "GMT": "Etc/GMT", "UTC": "UTC", "EUROPE/LONDON": "Europe/London",
        "LONDON": "Europe/London", "NEW YORK": "America/New_York",
        "NEW YORK TIME": "America/New_York", "TOKYO": "Asia/Tokyo",
        "SYDNEY": "Australia/Sydney",
    }
    upper = text.upper()
    for label, name in label_map.items():
        if label in upper:
            try:
                return name, ZoneInfo(name)
            except ZoneInfoNotFoundError:
                pass
    raise RuntimeError("Forex Factory page did not expose a complete supported calendar timezone")


def _find_cell(tr, field: str):
    """Prefer Forex Factory's canonical field selector, then an exact class token."""
    selector = f"td.calendar__cell.calendar__{field}.{field}"
    exact = tr.select_one(selector)
    if exact is not None:
        return exact
    wanted = f"calendar__{field}"
    for td in tr.find_all("td", recursive=False):
        if wanted in _class_tokens(td):
            return td
    return None


def _row_is_calendar(tr) -> bool:
    classes = _class_tokens(tr)
    return not classes or any("calendar__row" in c or "calendar_row" in c for c in classes)


def _numeric_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if number > 10_000_000_000:
        number /= 1000.0
    if number < 1_000_000_000 or number > 4_500_000_000:
        return None
    return datetime.fromtimestamp(number, tz=timezone.utc)


def row_source_timestamp(tr) -> datetime | None:
    attrs = (
        "data-timestamp", "data-event-timestamp", "data-time",
        "data-utc-timestamp", "data-release-timestamp",
    )
    for node in [tr, *tr.find_all(True)]:
        for attr in attrs:
            timestamp = _numeric_timestamp(node.get(attr))
            if timestamp is not None:
                return timestamp
    return None


def impact_label(cell) -> str:
    if cell is None:
        return ""
    candidates = []
    for node in cell.find_all(True):
        for attr in ("title", "aria-label", "data-title", "alt"):
            value = node.get(attr)
            if value:
                candidates.append(str(value))
        candidates.extend(str(c) for c in node.get("class", []))
    candidates.append(cell.get_text(" ", strip=True))
    text = " ".join(candidates).upper()
    if any(x in text for x in ("HIGH IMPACT", "IMPACT-RED", "IMPACT_RED", "FF-IMPACT-RED", "FF_IMPACT_RED")):
        return "HIGH"
    if any(x in text for x in ("MEDIUM IMPACT", "MED IMPACT", "IMPACT-ORANGE", "IMPACT_ORANGE", "IMPACT-YELLOW", "IMPACT_YELLOW", "FF-IMPACT-ORANGE", "FF-IMPACT-YELLOW")):
        return "MEDIUM"
    if any(x in text for x in ("LOW IMPACT", "IMPACT-GREY", "IMPACT_GRAY", "IMPACT-GRAY", "FF-IMPACT-GREY", "FF-IMPACT-GRAY")):
        return "LOW"
    return ""


def _visible_time_text(time_cell) -> str | None:
    """Return only the real release-time token from the canonical time cell.

    Forex Factory may include historical/reference nodes inside the same cell. We
    intentionally inspect the semantic <time> node first and only accept a strict
    standalone HH:MMam/pm token. No free-form cell text is ever passed to strptime.
    """
    if time_cell is None:
        return None

    nodes = time_cell.find_all("time")
    candidates = []
    for node in nodes:
        for value in (node.get("datetime"), node.get_text(" ", strip=True)):
            if value:
                candidates.append(value)

    for value in candidates:
        match = TIME_RE.fullmatch(" ".join(str(value).split()))
        if match:
            return f"{int(match.group('hour'))}:{match.group('minute') or '00'}{match.group('ampm').lower()}"

    direct_text = time_cell.get_text(" ", strip=True)
    for token in re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", direct_text, flags=re.IGNORECASE):
        normalized = " ".join(token.split())
        match = TIME_RE.fullmatch(normalized)
        if match:
            return f"{int(match.group('hour'))}:{match.group('minute') or '00'}{match.group('ampm').lower()}"
    return None


def _visible_date_text(date_cell) -> str | None:
    """Return only a strict weekday/month/day date token from the date cell."""
    if date_cell is None:
        return None
    nodes = date_cell.find_all(True)
    candidates = []
    for node in nodes:
        for attr in ("datetime", "data-date"):
            value = node.get(attr)
            if value:
                candidates.append(value)
        text = node.get_text(" ", strip=True)
        if text:
            candidates.append(text)
    candidates.append(date_cell.get_text(" ", strip=True))

    for value in candidates:
        normalized = " ".join(str(value).split())
        if DATE_RE.fullmatch(normalized):
            return normalized
    return None


def parse_local_event_datetime(current_date: date, time_text: str, source_tz: ZoneInfo) -> datetime:
    match = TIME_RE.fullmatch(" ".join(str(time_text).split()))
    if not match:
        raise ValueError(f"Ambiguous/invalid Forex Factory time cell: {time_text!r}")
    normalized = f"{int(match.group('hour'))}:{match.group('minute') or '00'}{match.group('ampm').lower()}"
    naive = datetime.strptime(f"{current_date} {normalized}", "%Y-%m-%d %I:%M%p")
    return naive.replace(tzinfo=source_tz)


def scrape_week(day: date, session: requests.Session) -> list[dict]:
    url = f"https://www.forexfactory.com/calendar?week={week_key(day)}"
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", class_=lambda value: value and "calendar__table" in value)
            if table is None:
                raise RuntimeError(f"Calendar table not found: {url}")
            source_timezone_name, source_tz = extract_page_timezone(soup)

            rows = []
            current_date = None
            raw_calendar_rows = 0
            raw_eur_usd_rows = 0
            for tr in table.find_all("tr"):
                if not _row_is_calendar(tr):
                    continue
                raw_calendar_rows += 1
                date_cell = _find_cell(tr, "date")
                time_cell = _find_cell(tr, "time")
                date_text = _visible_date_text(date_cell)
                if date_text:
                    resolved = resolve_calendar_date(date_text, day)
                    if resolved is not None:
                        current_date = resolved

                currency_cell = _find_cell(tr, "currency")
                event_cell = _find_cell(tr, "event")
                if not (current_date and currency_cell and event_cell):
                    continue
                currency = currency_cell.get_text(" ", strip=True).upper()
                if currency not in {"EUR", "USD"}:
                    continue
                raw_eur_usd_rows += 1

                impact_cell = _find_cell(tr, "impact")
                actual_cell = _find_cell(tr, "actual")
                forecast_cell = _find_cell(tr, "forecast")
                previous_cell = _find_cell(tr, "previous")
                impact = impact_label(impact_cell) or "LOW"
                event = event_cell.get_text(" ", strip=True)
                actual = actual_cell.get_text(" ", strip=True) if actual_cell else ""
                forecast = forecast_cell.get_text(" ", strip=True) if forecast_cell else ""
                previous = previous_cell.get_text(" ", strip=True) if previous_cell else ""

                time_text = _visible_time_text(time_cell)
                if time_text is None:
                    raise RuntimeError(f"EUR/USD row has no unambiguous release time in canonical time cell: {url}")

                source_dt = row_source_timestamp(tr)
                if source_dt is None:
                    source_dt = parse_local_event_datetime(current_date, time_text, source_tz)
                    release_utc = source_dt.astimezone(UTC)
                else:
                    release_utc = source_dt.astimezone(UTC)
                    source_dt = release_utc.astimezone(source_tz)

                rows.append({
                    "release_time": release_utc.isoformat(),
                    "source_timezone": source_timezone_name,
                    "source_date": source_dt.date().isoformat(),
                    "source_local_time": source_dt.strftime("%H:%M:%S"),
                    "currency": currency,
                    "impact": impact,
                    "event": event,
                    "bias": compute_bias(currency, event, actual, forecast),
                    "actual": actual,
                    "forecast": forecast,
                    "previous": previous,
                })

            if not rows and raw_eur_usd_rows:
                raise RuntimeError(f"Found {raw_eur_usd_rows} EUR/USD rows but parsed 0 events: {url}")
            if not rows and raw_calendar_rows == 0:
                raise RuntimeError(f"Calendar table found but 0 calendar rows matched: {url}")
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
                failures.append({"week_start": cursor.isoformat(), "week_key": week_key(cursor), "error": str(exc)})
            cursor += timedelta(days=7)

    if failures:
        FAILURES.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["week_start", "week_key", "error"]).to_csv(FAILURES, index=False)
        raise RuntimeError(f"Historical calendar fetch incomplete: {len(failures)}/{requested_weeks} weeks failed; refusing to publish partial Step 10 data")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No EUR/USD economic events were downloaded")

    df = df.drop_duplicates(subset=["release_time", "currency", "event"])
    df["release_time"] = pd.to_datetime(df["release_time"], utc=True)
    df = df.sort_values("release_time")

    if df["source_timezone"].isna().any() or df["source_timezone"].eq("").any():
        raise RuntimeError("One or more events have no source timezone provenance")
    if not df["release_time"].is_monotonic_increasing:
        raise RuntimeError("Economic calendar release_time is not monotonic after normalization")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    FETCH_STATS.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = OUT.with_suffix(".tmp")
    df.to_csv(tmp_out, index=False)
    tmp_out.replace(OUT)

    pd.DataFrame([], columns=["week_start", "week_key", "error"]).to_csv(FAILURES, index=False)
    pd.DataFrame([{
        "price_start": start.isoformat(),
        "price_end": end.isoformat(),
        "requested_weeks": requested_weeks,
        "successful_weeks": requested_weeks,
        "failed_weeks": 0,
        "event_count": len(df),
        "timezone_count": df["source_timezone"].nunique(),
    }]).to_csv(FETCH_STATS, index=False)

    print(f"Saved {len(df)} EUR/USD events to {OUT}")
    print(f"Weeks: {requested_weeks} successful, 0 failed")


if __name__ == "__main__":
    main()
