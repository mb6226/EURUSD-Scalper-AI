from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data/eurusd/EURUSD_1m.csv"
SELECTED = ROOT / "results/step12_economic_filtered_stats/step12_162_signal_list.csv"
STEP7 = ROOT / "results/zigzag_step7/swings_with_previous_fibo.csv"
OUT = ROOT / "results/step13_price_action"
OUT.mkdir(parents=True, exist_ok=True)

PIP = 0.0001
FIB_RATIOS = [78.6, 100.0, 127.2, 161.8, 200.0]


def load_price() -> pd.DataFrame:
    df = pd.read_csv(PRICE)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price data missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def load_selected() -> pd.DataFrame:
    sel = pd.read_csv(SELECTED)
    step7 = pd.read_csv(STEP7)
    required = {"swing_id", "direction", "trend", "swing_pips", "bars", "prior_direction", "prior_swing_pips"}
    missing = required - set(sel.columns)
    if missing:
        raise ValueError(f"Step 12 signal list missing columns: {sorted(missing)}")
    cols = ["swing_id", "start_bar", "end_bar", "start_time", "end_time", "start_price", "end_price", "direction", "swing_pips", "bars", "prior_direction", "prior_swing_pips", "prior_start_price", "prior_end_price"]
    missing7 = set(cols) - set(step7.columns)
    if missing7:
        raise ValueError(f"Step 7 data missing columns: {sorted(missing7)}")
    return sel.merge(step7[cols], on="swing_id", how="inner", validate="one_to_one", suffixes=("", "_step7"))


def fib_zone(x: float) -> str:
    edges = [-np.inf, 0, 23.6, 38.2, 50, 61.8, 78.6, 100, 127.2, 161.8, 200, 261.8, 361.8, 423.6, np.inf]
    labels = ["<0", "0-23.6", "23.6-38.2", "38.2-50", "50-61.8", "61.8-78.6", "78.6-100", "100-127.2", "127.2-161.8", "161.8-200", "200-261.8", "261.8-361.8", "361.8-423.6", ">=423.6"]
    i = np.searchsorted(edges, x, side="right") - 1
    return labels[max(0, min(i, len(labels) - 1))]


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["range_pips"] = (x["high"] - x["low"]) / PIP
    x["body_pips"] = (x["close"] - x["open"]).abs() / PIP
    x["body_signed_pips"] = (x["close"] - x["open"]) / PIP
    x["upper_wick_pips"] = (x["high"] - x[["open", "close"]].max(axis=1)) / PIP
    x["lower_wick_pips"] = (x[["open", "close"]].min(axis=1) - x["low"]) / PIP
    x["body_ratio"] = x["body_pips"] / x["range_pips"].replace(0, np.nan)
    x["close_location"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
    x["return_1m_pips"] = x["close"].diff() / PIP
    x["return_3m_pips"] = x["close"].diff(3) / PIP
    x["return_5m_pips"] = x["close"].diff(5) / PIP
    x["return_15m_pips"] = x["close"].diff(15) / PIP
    for n in [5, 10, 20, 50]:
        prior_high = x["high"].shift(1).rolling(n, min_periods=n).max()
        prior_low = x["low"].shift(1).rolling(n, min_periods=n).min()
        x[f"prior_{n}m_high"] = prior_high
        x[f"prior_{n}m_low"] = prior_low
        x[f"break_above_{n}m"] = x["close"] > prior_high
        x[f"break_below_{n}m"] = x["close"] < prior_low
        x[f"dist_prior_{n}m_high_pips"] = (prior_high - x["close"]) / PIP
        x[f"dist_prior_{n}m_low_pips"] = (x["close"] - prior_low) / PIP
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - x["close"].shift()).abs(),
        (x["low"] - x["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    x["atr14_pips"] = tr.rolling(14, min_periods=14).mean() / PIP
    x["range_vs_atr14"] = x["range_pips"] / x["atr14_pips"].replace(0, np.nan)
    x["body_vs_atr14"] = x["body_pips"] / x["atr14_pips"].replace(0, np.nan)
    x["bullish_candle"] = x["close"] > x["open"]
    x["bearish_candle"] = x["close"] < x["open"]
    x["strong_bullish"] = (x["body_ratio"] >= 0.60) & (x["close_location"] >= 0.70)
    x["strong_bearish"] = (x["body_ratio"] >= 0.60) & (x["close_location"] <= 0.30)
    x["bullish_engulf_prev"] = (x["close"] > x["open"]) & (x["open"] <= x["close"].shift(1)) & (x["close"] >= x["open"].shift(1)) & (x["open"].shift(1) > x["close"].shift(1))
    x["bearish_engulf_prev"] = (x["close"] < x["open"]) & (x["open"] >= x["close"].shift(1)) & (x["close"] <= x["open"].shift(1)) & (x["open"].shift(1) < x["close"].shift(1))
    return x


def build() -> None:
    price = add_price_features(load_price())
    selected = load_selected()
    rows = []
    summary = []

    for _, s in selected.sort_values("swing_id").iterrows():
        start = int(s["start_bar"])
        end = int(s["end_bar"])
        if start < 0 or end >= len(price) or start > end:
            raise ValueError(f"Invalid swing bar range for swing {s['swing_id']}: {start}-{end}")
        seg = price.iloc[start:end + 1].copy()
        seg["swing_id"] = int(s["swing_id"])
        seg["swing_direction"] = s["direction"]
        seg["trend"] = s["trend"]
        seg["economic_regime"] = s.get("economic_regime", "")
        seg["prior_direction"] = s["prior_direction"]
        seg["swing_pips"] = float(s["swing_pips"])
        seg["prior_swing_pips"] = float(s["prior_swing_pips"])
        seg["swing_bar_index"] = np.arange(len(seg))
        seg["swing_position_pct"] = 100.0 * seg["swing_bar_index"] / max(1, len(seg) - 1)
        seg["bars_remaining_to_endpoint"] = len(seg) - 1 - seg["swing_bar_index"]

        prior_start = float(s["prior_start_price"])
        prior_end = float(s["prior_end_price"])
        prior_range = abs(prior_end - prior_start)
        if s["prior_direction"] == "UP":
            reaction_price = seg["low"]
            retr = (prior_end - reaction_price) / prior_range * 100.0 if prior_range else np.nan
        elif s["prior_direction"] == "DOWN":
            reaction_price = seg["high"]
            retr = (reaction_price - prior_end) / prior_range * 100.0 if prior_range else np.nan
        else:
            reaction_price = np.nan
            retr = np.nan
        seg["prior_fibo_retracement_pct"] = retr
        seg["prior_fibo_zone"] = [fib_zone(v) if pd.notna(v) else "" for v in retr]
        for ratio in FIB_RATIOS:
            if s["prior_direction"] == "UP":
                level = prior_end - prior_range * ratio / 100.0
            else:
                level = prior_end + prior_range * ratio / 100.0
            seg[f"dist_fib_{str(ratio).replace('.', '_')}_pips"] = (seg["close"] - level).abs() / PIP
        seg["price_direction_alignment"] = ((s["direction"] == "UP") & (seg["close"] > seg["open"])) | ((s["direction"] == "DOWN") & (seg["close"] < seg["open"]))
        rows.append(seg)

        summary.append({
            "swing_id": int(s["swing_id"]),
            "direction": s["direction"],
            "trend": s["trend"],
            "economic_regime": s.get("economic_regime", ""),
            "bars": len(seg),
            "swing_pips": float(s["swing_pips"]),
            "median_range_pips": seg["range_pips"].median(),
            "median_body_pips": seg["body_pips"].median(),
            "mean_body_ratio": seg["body_ratio"].mean(),
            "bullish_candle_pct": (seg["bullish_candle"]).mean(),
            "strong_directional_candle_pct": ((seg["strong_bullish"] if s["direction"] == "UP" else seg["strong_bearish"]).mean()),
            "break_above_20m_pct": seg["break_above_20m"].mean(),
            "break_below_20m_pct": seg["break_below_20m"].mean(),
            "median_atr14_pips": seg["atr14_pips"].median(),
            "max_prior_fibo_retracement_pct": seg["prior_fibo_retracement_pct"].max(),
            "endpoint_fibo_zone": seg["prior_fibo_zone"].iloc[-1],
        })

    bars = pd.concat(rows, ignore_index=True)
    signal_summary = pd.DataFrame(summary)
    bars.to_csv(OUT / "step13_162_price_action_bars.csv", index=False)
    signal_summary.to_csv(OUT / "step13_signal_price_action_summary.csv", index=False)

    overall = pd.DataFrame([{
        "selected_signals": len(signal_summary),
        "price_action_bars": len(bars),
        "up_signals": int((signal_summary["direction"] == "UP").sum()),
        "down_signals": int((signal_summary["direction"] == "DOWN").sum()),
        "median_swing_pips": signal_summary["swing_pips"].median(),
        "mean_swing_pips": signal_summary["swing_pips"].mean(),
        "median_swing_bars": signal_summary["bars"].median(),
        "median_atr14_pips": signal_summary["median_atr14_pips"].median(),
    }])
    overall.to_csv(OUT / "step13_overall_summary.csv", index=False)

    print("Step 13 — M1 Price Action Feature Extraction")
    print(overall.to_string(index=False))
    print("Outputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    build()
