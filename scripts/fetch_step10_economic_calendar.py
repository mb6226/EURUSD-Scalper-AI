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
TIME_RE = re.compile(r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)$", re.I)
TIME_TOKEN_RE = re.compile(r"(?<!\d)(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)(?![A-Za-z])", re.I)
DATE_RE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?$", re.I)


def parse_number(value: str):
    if value is None: return None
    text = str(value).replace(",", "").strip()
    if not text or text.lower() in {"nan", "none", "n/a", "-"}: return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not m: return None
    number = float(m.group(0)); suffix = text[m.end():].strip().upper()
    if suffix.startswith("K"): number *= 1_000
    elif suffix.startswith("M"): number *= 1_000_000
    elif suffix.startswith("B"): number *= 1_000_000_000
    return number


def higher_is_better(event: str) -> bool:
    name = event.lower()
    return not any(x in name for x in ("unemployment rate", "unemployment change", "jobless claims", "initial claims", "continuing claims", "job cuts", "unemployed"))


def compute_bias(currency: str, event: str, actual: str, forecast: str) -> int:
    a, f = parse_number(actual), parse_number(forecast)
    if a is None or f is None or a == f: return 0
    good = (a > f) if higher_is_better(event) else (a < f)
    sign = 1 if good else -1
    return sign if currency == "EUR" else -sign


def price_range() -> tuple[date, date]:
    px = pd.read_csv(PRICE, usecols=["timestamp"])
    ts = pd.to_datetime(px["timestamp"], errors="coerce").dropna()
    if ts.empty: raise ValueError("EURUSD price file has no valid timestamps")
    return ts.dt.date.min(), ts.dt.date.max()


def week_key(day: date) -> str:
    return f"{day.strftime('%b').lower()}{day.day}.{day.year}"


def resolve_calendar_date(text: str, week_day: date) -> date | None:
    m = DATE_RE.fullmatch(" ".join(str(text).split()))
    if not m: return None
    month_num = datetime.strptime(m.group("month"), "%b").month
    year = week_day.year
    if week_day.month == 12 and month_num == 1: year += 1
    elif week_day.month == 1 and month_num == 12: year -= 1
    try: return date(year, month_num, int(m.group("day")))
    except ValueError: return None


def _class_tokens(tag) -> set[str]:
    if tag is None: return set()
    classes = tag.get("class", [])
    if isinstance(classes, str): classes = classes.split()
    return {str(x) for x in classes}


def extract_page_timezone(soup: BeautifulSoup) -> tuple[str, ZoneInfo]:
    text = soup.get_text(" ", strip=True)
    for pattern in (r"Calendar\s+Time\s+Zone\s*:\s*([A-Za-z_]+/[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+)*)", r"Time\s+Zone\s*:\s*([A-Za-z_]+/[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+)*)"):
        for name in re.findall(pattern, text, flags=re.I):
            try: return name, ZoneInfo(name)
            except ZoneInfoNotFoundError: pass
    labels = {"GMT":"Etc/GMT", "UTC":"UTC", "EUROPE/LONDON":"Europe/London", "LONDON":"Europe/London", "NEW YORK":"America/New_York", "NEW YORK TIME":"America/New_York", "TOKYO":"Asia/Tokyo", "SYDNEY":"Australia/Sydney"}
    upper = text.upper()
    for label, name in labels.items():
        if label in upper:
            try: return name, ZoneInfo(name)
            except ZoneInfoNotFoundError: pass
    raise RuntimeError("Forex Factory page did not expose a complete supported calendar timezone")


def _find_cell(tr, field: str):
    # Forex Factory has rendered calendar fields as both <td> cells and
    # nested elements across historical/current layouts. Prefer the semantic
    # field class anywhere in the row; fall back to an exact class-token scan.
    for selector in (f".calendar__cell.calendar__{field}.{field}", f".calendar__{field}.{field}"):
        exact = tr.select_one(selector)
        if exact is not None: return exact
    wanted = f"calendar__{field}"
    for node in tr.find_all(True):
        if wanted in _class_tokens(node): return node
    return None


def _row_is_calendar(tr) -> bool:
    classes = _class_tokens(tr)
    return not classes or any("calendar__row" in c or "calendar_row" in c for c in classes)


def _numeric_timestamp(value: str | None) -> datetime | None:
    if value is None: return None
    try: number = float(str(value).strip())
    except ValueError: return None
    if number > 10_000_000_000: number /= 1000.0
    if number < 1_000_000_000 or number > 4_500_000_000: return None
    return datetime.fromtimestamp(number, tz=timezone.utc)


def row_source_timestamp(tr) -> datetime | None:
    attrs = ("data-timestamp", "data-event-timestamp", "data-time", "data-utc-timestamp", "data-release-timestamp")
    for node in [tr, *tr.find_all(True)]:
        for attr in attrs:
            ts = _numeric_timestamp(node.get(attr))
            if ts is not None: return ts
    return None


def impact_label(cell) -> str:
    if cell is None: return ""
    candidates = []
    for node in cell.find_all(True):
        for attr in ("title", "aria-label", "data-title", "alt"):
            if node.get(attr): candidates.append(str(node.get(attr)))
        candidates.extend(str(c) for c in node.get("class", []))
    candidates.append(cell.get_text(" ", strip=True)); text = " ".join(candidates).upper()
    if any(x in text for x in ("HIGH IMPACT", "IMPACT-RED", "IMPACT_RED", "FF-IMPACT-RED", "FF_IMPACT_RED")): return "HIGH"
    if any(x in text for x in ("MEDIUM IMPACT", "MED IMPACT", "IMPACT-ORANGE", "IMPACT_ORANGE", "IMPACT-YELLOW", "IMPACT_YELLOW", "FF-IMPACT-ORANGE", "FF-IMPACT-YELLOW")): return "MEDIUM"
    if any(x in text for x in ("LOW IMPACT", "IMPACT-GREY", "IMPACT_GRAY", "IMPACT-GRAY", "FF-IMPACT-GREY", "FF-IMPACT-GRAY")): return "LOW"
    return ""


def _canonical_time_text(time_cell) -> str | None:
    """Extract exactly one release-time token from the canonical time field.

    The field may contain plain text such as ``8:30am`` or nested markup.
    Non-release text such as ``Sep Data`` / ``Nov 15th`` is ignored. If the
    same canonical field exposes two different exact times, fail closed.
    """
    if time_cell is None: return None
    candidates: list[str] = []

    def collect(value) -> None:
        if not value: return
        text = " ".join(str(value).split())
        for match in TIME_TOKEN_RE.finditer(text):
            candidates.append(match.group(0))

    for node in [time_cell, *time_cell.find_all(True)]:
        for attr in ("datetime", "data-time", "data-event-time"):
            collect(node.get(attr))
        collect(node.get_text(" ", strip=True))

    normalized = set()
    for value in candidates:
        match = TIME_TOKEN_RE.fullmatch(" ".join(value.split()))
        if match is None: continue
        normalized.add(f"{int(match.group('hour'))}:{match.group('minute') or '00'}{match.group('ampm').lower()}")
    if len(normalized) == 1: return next(iter(normalized))
    if len(normalized) > 1: raise RuntimeError("Canonical Forex Factory time cell contains conflicting exact release times")
    return None


def _canonical_date_text(date_cell) -> str | None:
    if date_cell is None: return None
    direct = " ".join(date_cell.get_text(" ", strip=True).split())
    if DATE_RE.fullmatch(direct): return direct
    for node in date_cell.find_all(True):
        for value in (node.get("datetime"), node.get("data-date"), node.get_text(" ", strip=True)):
            if value and DATE_RE.fullmatch(" ".join(str(value).split())): return " ".join(str(value).split())
    return None


def parse_local_event_datetime(current_date: date, time_text: str, source_tz: ZoneInfo) -> datetime:
    m = TIME_RE.fullmatch(" ".join(str(time_text).split()))
    if not m: raise ValueError(f"Invalid canonical Forex Factory time: {time_text!r}")
    normalized = f"{int(m.group('hour'))}:{m.group('minute') or '00'}{m.group('ampm').lower()}"
    return datetime.strptime(f"{current_date} {normalized}", "%Y-%m-%d %I:%M%p").replace(tzinfo=source_tz)


def scrape_week(day: date, session: requests.Session) -> tuple[list[dict], int]:
    url = f"https://www.forexfactory.com/calendar?week={week_key(day)}"
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=HEADERS, timeout=30); response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", class_=lambda value: value and "calendar__table" in value)
            if table is None: raise RuntimeError(f"Calendar table not found: {url}")
            source_timezone_name, source_tz = extract_page_timezone(soup)
            rows, current_date, raw_calendar_rows, raw_eur_usd_rows, skipped_untimed = [], None, 0, 0, 0
            for tr in table.find_all("tr"):
                if not _row_is_calendar(tr): continue
                raw_calendar_rows += 1
                date_text = _canonical_date_text(_find_cell(tr, "date"))
                if date_text:
                    resolved = resolve_calendar_date(date_text, day)
                    if resolved is not None: current_date = resolved
                currency_cell, event_cell = _find_cell(tr, "currency"), _find_cell(tr, "event")
                if not (current_date and currency_cell and event_cell): continue
                currency = currency_cell.get_text(" ", strip=True).upper()
                if currency not in {"EUR", "USD"}: continue
                raw_eur_usd_rows += 1
                time_text = _canonical_time_text(_find_cell(tr, "time"))
                if time_text is None:
                    skipped_untimed += 1
                    continue
                impact = impact_label(_find_cell(tr, "impact")) or "LOW"
                event = event_cell.get_text(" ", strip=True)
                actual_cell, forecast_cell, previous_cell = _find_cell(tr, "actual"), _find_cell(tr, "forecast"), _find_cell(tr, "previous")
                actual = actual_cell.get_text(" ", strip=True) if actual_cell else ""
                forecast = forecast_cell.get_text(" ", strip=True) if forecast_cell else ""
                previous = previous_cell.get_text(" ", strip=True) if previous_cell else ""
                source_dt = row_source_timestamp(tr)
                if source_dt is None: source_dt = parse_local_event_datetime(current_date, time_text, source_tz)
                release_utc = source_dt.astimezone(UTC); source_dt = release_utc.astimezone(source_tz)
                rows.append({"release_time": release_utc.isoformat(), "source_timezone": source_timezone_name, "source_date": source_dt.date().isoformat(), "source_local_time": source_dt.strftime("%H:%M:%S"), "currency": currency, "impact": impact, "event": event, "bias": compute_bias(currency, event, actual, forecast), "actual": actual, "forecast": forecast, "previous": previous})
            if not rows and raw_eur_usd_rows and skipped_untimed == 0: raise RuntimeError(f"Found {raw_eur_usd_rows} EUR/USD rows but parsed 0 events: {url}")
            if not rows and raw_calendar_rows == 0: raise RuntimeError(f"Calendar table found but 0 calendar rows matched: {url}")
            return rows, skipped_untimed
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_SECONDS[attempt]; print(f"  attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}; retrying in {delay}s"); time.sleep(delay)
    raise RuntimeError(last_error or f"Unknown fetch error: {url}")


def main() -> None:
    start, end = price_range(); cursor = start - timedelta(days=(start.weekday() + 1) % 7)
    rows, failures, requested_weeks, skipped_untimed_total = [], [], 0, 0
    with requests.Session() as session:
        while cursor <= end:
            requested_weeks += 1; print(f"Fetching economic calendar week {cursor}")
            try:
                week_rows, skipped = scrape_week(cursor, session); rows.extend(week_rows); skipped_untimed_total += skipped
                print(f"  success: {len(week_rows)} exact-time EUR/USD events; skipped {skipped} untimed/non-release rows")
            except Exception as exc:
                print(f"  FAILED: {exc}"); failures.append({"week_start": cursor.isoformat(), "week_key": week_key(cursor), "error": str(exc)})
            cursor += timedelta(days=7)
    FAILURES.parent.mkdir(parents=True, exist_ok=True); FETCH_STATS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["week_start", "week_key", "error"]).to_csv(FAILURES, index=False)
    if failures: raise RuntimeError(f"Historical calendar fetch incomplete: {len(failures)}/{requested_weeks} weeks failed; refusing to publish partial Step 10 data")
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No EUR/USD economic events with exact release times were downloaded")
    df = df.drop_duplicates(subset=["release_time", "currency", "event"]); df["release_time"] = pd.to_datetime(df["release_time"], utc=True); df = df.sort_values("release_time")
    if df["source_timezone"].isna().any() or df["source_timezone"].eq("").any(): raise RuntimeError("One or more events have no source timezone provenance")
    if not df["release_time"].is_monotonic_increasing: raise RuntimeError("Economic calendar release_time is not monotonic after normalization")
    OUT.parent.mkdir(parents=True, exist_ok=True); tmp_out = OUT.with_suffix(".tmp"); df.to_csv(tmp_out, index=False); tmp_out.replace(OUT)
    pd.DataFrame([{"price_start": start.isoformat(), "price_end": end.isoformat(), "requested_weeks": requested_weeks, "successful_weeks": requested_weeks, "failed_weeks": 0, "skipped_untimed_rows": skipped_untimed_total, "event_count": len(df), "timezone_count": df["source_timezone"].nunique()}]).to_csv(FETCH_STATS, index=False)


if __name__ == "__main__": main()
