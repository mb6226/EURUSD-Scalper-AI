from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.zigzag_mt5 import zigzag_mt5_with_point

RESULT_DIR = Path("results/zigzag_structural")
DATA_PATH = Path("data/eurusd/EURUSD_1m.csv")
STEP3 = Path("results/zigzag_quarterly/quarter_top20.csv")
EURUSD_POINT = 0.00001


def pick_configs():
    df = pd.read_csv(STEP3)
    depths = sorted(df["depth"].dropna().astype(int).unique())
    deviations = sorted(df["deviation"].dropna().astype(int).unique())
    backsteps = sorted(df["backstep"].dropna().astype(int).unique())
    if not depths or not deviations or not backsteps:
        raise RuntimeError("Step 3 candidate file is empty")
    center = int(round(float(df["depth"].median()) / 5) * 5)
    test_depths = sorted(set([center - 5, center, center + 5]))
    return [(d, v, b) for d in test_depths for v in deviations for b in backsteps]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    rename = {lower[x]: x for x in ("open", "high", "low", "close") if x in lower}
    if len(rename) != 4:
        raise ValueError(f"Missing OHLC columns in {path}")
    time_col = next((lower[x] for x in ("timestamp", "datetime", "time", "date") if x in lower), None)
    if time_col is None:
        raise ValueError(f"Missing timestamp column in {path}")
    df = df.rename(columns=rename)
    df.index = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[df.index.notna()].sort_index()
    return df[["open", "high", "low", "close"]]


def pivots_from_zigzag(data: pd.DataFrame, depth: int, deviation: int, backstep: int) -> pd.DataFrame:
    zz = zigzag_mt5_with_point(
        data["high"].to_numpy(),
        data["low"].to_numpy(),
        depth=depth,
        deviation_points=deviation,
        backstep=backstep,
        point=EURUSD_POINT,
    )
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    rows = []
    for i, value in enumerate(zz):
        if not math.isfinite(float(value)):
            continue
        kind = "high" if abs(float(value) - highs[i]) <= abs(float(value) - lows[i]) else "low"
        rows.append((i, data.index[i], float(value), kind))
    return pd.DataFrame(rows, columns=["bar", "timestamp", "price", "kind"])


def structural_metrics(data: pd.DataFrame, pivots: pd.DataFrame) -> dict[str, float]:
    names = [
        "pivots", "swings", "median_swing_pips", "median_bars",
        "trend_follow_rate", "reversal_follow_rate", "false_reversal_rate",
        "continuation_rate", "trend_rate", "range_rate",
        "trend_efficiency", "range_efficiency", "m5_alignment_rate",
    ]
    if len(pivots) < 4:
        return {k: float("nan") for k in names}

    close = data["close"].to_numpy()
    swings = []
    for a, b in zip(pivots.iloc[:-1].itertuples(), pivots.iloc[1:].itertuples()):
        bars = int(b.bar - a.bar)
        if bars <= 0:
            continue
        move_pips = abs(b.price - a.price) * 10000.0
        path = float(pd.Series(close[int(a.bar):int(b.bar) + 1]).diff().abs().sum())
        efficiency = move_pips / (path * 10000.0) if path > 0 else 1.0
        direction = 1 if b.price > a.price else -1
        swings.append((a.bar, b.bar, bars, move_pips, efficiency, direction))

    s = pd.DataFrame(
        swings,
        columns=["a", "b", "bars", "swing_pips", "efficiency", "direction"],
    )
    if s.empty:
        return {k: float("nan") for k in names}

    same = s["direction"].shift(1) == s["direction"]
    valid = same.iloc[1:]
    reversal = ~valid
    prev_size = s["swing_pips"].shift(1).iloc[1:].to_numpy()
    cur_size = s["swing_pips"].iloc[1:].to_numpy()
    false_rev = reversal.to_numpy() & (cur_size < prev_size * 0.5)
    continuation = float(valid.mean())
    false_reversal = float(false_rev.mean())

    roll = s["direction"].rolling(10, min_periods=5).mean().abs()
    trend_mask = roll >= 0.5
    range_mask = roll < 0.5

    m5c = data["close"].resample("5min").last().dropna()
    aligned = []
    for r in s.itertuples():
        ca = m5c.asof(pd.Timestamp(data.index[int(r.a)]))
        cb = m5c.asof(pd.Timestamp(data.index[int(r.b)]))
        if pd.isna(ca) or pd.isna(cb) or cb == ca:
            continue
        aligned.append((1 if cb > ca else -1) == r.direction)

    return {
        "pivots": float(len(pivots)),
        "swings": float(len(s)),
        "median_swing_pips": float(s["swing_pips"].median()),
        "median_bars": float(s["bars"].median()),
        "trend_follow_rate": float(1.0 - false_reversal),
        "reversal_follow_rate": float(reversal.mean()),
        "false_reversal_rate": false_reversal,
        "continuation_rate": continuation,
        "trend_rate": float(trend_mask.mean()),
        "range_rate": float(range_mask.mean()),
        "trend_efficiency": float(s.loc[trend_mask, "efficiency"].median()) if trend_mask.any() else float("nan"),
        "range_efficiency": float(s.loc[range_mask, "efficiency"].median()) if range_mask.any() else float("nan"),
        "m5_alignment_rate": float(sum(aligned) / len(aligned)) if aligned else float("nan"),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    if not STEP3.exists():
        raise FileNotFoundError(STEP3)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(DATA_PATH)
    configs = pick_configs()
    rows = []
    for depth, deviation, backstep in configs:
        pivots = pivots_from_zigzag(data, depth, deviation, backstep)
        rows.append(
            {
                "depth": depth,
                "deviation": deviation,
                "backstep": backstep,
                **structural_metrics(data, pivots),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "structural_all_results.csv", index=False)

    numeric = [c for c in out.columns if c not in {"depth", "deviation", "backstep"}]
    out.groupby("depth", as_index=False)[numeric].median().to_csv(
        RESULT_DIR / "structural_depth_summary.csv", index=False
    )

    regime_cols = [
        "trend_follow_rate", "reversal_follow_rate", "false_reversal_rate",
        "continuation_rate", "trend_rate", "range_rate",
        "trend_efficiency", "range_efficiency", "m5_alignment_rate",
    ]
    out.groupby("depth", as_index=False)[regime_cols].median().to_csv(
        RESULT_DIR / "regime_summary.csv", index=False
    )

    out.groupby("depth").agg(
        configs=("depth", "size"),
        median_false_reversal=("false_reversal_rate", "median"),
        max_false_reversal=("false_reversal_rate", "max"),
        median_m5_alignment=("m5_alignment_rate", "median"),
        min_m5_alignment=("m5_alignment_rate", "min"),
        median_continuation=("continuation_rate", "median"),
        median_trend_rate=("trend_rate", "median"),
        median_range_rate=("range_rate", "median"),
    ).reset_index().to_csv(RESULT_DIR / "structural_robustness.csv", index=False)

    (RESULT_DIR / "README.md").write_text(
        "# Step 5 — ZigZag Structural Regime Validation\n\n"
        "Consumes Step 3 quarterly candidates and validates structural behavior "
        "around Depth≈350. Uses the MT5-compatible ZigZag core with EURUSD "
        "point size 0.00001. Descriptive only; no entry rule or PnL optimization.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
