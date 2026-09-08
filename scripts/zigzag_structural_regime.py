from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.zigzag_mt5 import zigzag_mt5

RESULT_DIR = Path("results/zigzag_structural")
DATA_PATH = Path("data/eurusd/EURUSD_1m.csv")
STEP3 = Path("results/zigzag_quarterly/quarter_top20.csv")


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
