from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data/eurusd/EURUSD_1m.csv"
SELECTED = ROOT / "results/step12_economic_filtered_stats/step12_162_signal_list.csv"
STEP7 = ROOT / "results/zigzag_step7/swings_with_previous_fibo.csv"
OUT = ROOT / "results/step14_price_action_event_study"
OUT.mkdir(parents=True, exist_ok=True)
PIP = 0.0001

BREAK_LOOKBACK = 20
RETEST_LOOKBACK = 10
RETEST_ATR_MULT = 0.25
FIB_ATR_TOL = 0.25
STRUCTURE_LOOKBACK = 3
STRUCTURE_WINDOW = 10
MOM_BODY_RATIO = 0.60
MOM_RANGE_ATR = 1.20
MOM_RETURN_3M = 0.50
FIB_LEVELS = (78.6, 100.0, 127.2, 161.8, 200.0)


def load_price() -> pd.DataFrame:
    x = pd.read_csv(PRICE)
    req = {"timestamp", "open", "high", "low", "close"}
    missing = req - set(x.columns)
    if missing:
        raise ValueError(f"Price data missing columns: {sorted(missing)}")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)


def load_selected() -> pd.DataFrame:
    sel = pd.read_csv(SELECTED)
    step7 = pd.read_csv(STEP7)
    req = {"swing_id", "direction", "trend", "swing_pips", "bars", "prior_direction", "prior_swing_pips"}
    if (m := req - set(sel.columns)):
        raise ValueError(f"Step 12 signal list missing columns: {sorted(m)}")
    # Step 12 is canonical for signal metadata. Step 7 supplies only coordinates.
    cols = ["swing_id", "start_bar", "end_bar", "start_time", "end_time", "start_price", "end_price",
            "prior_start_price", "prior_end_price"]
    if (m := set(cols) - set(step7.columns)):
        raise ValueError(f"Step 7 data missing columns: {sorted(m)}")
    return sel.merge(step7[cols], on="swing_id", how="inner", validate="one_to_one")


def features(seg: pd.DataFrame) -> pd.DataFrame:
    x = seg.copy().reset_index(drop=True)
    x["range_pips"] = (x.high - x.low) / PIP
    x["body_pips"] = (x.close - x.open).abs() / PIP
    x["body_ratio"] = x.body_pips / x.range_pips.replace(0, np.nan)
    x["close_location"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
    x["upper_wick_pips"] = (x.high - x[["open", "close"]].max(axis=1)) / PIP
    x["lower_wick_pips"] = (x[["open", "close"]].min(axis=1) - x.low) / PIP
    x["return_3m_pips"] = x.close.diff(3) / PIP
    tr = pd.concat([x.high-x.low, (x.high-x.close.shift()).abs(), (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    x["atr14_pips"] = tr.rolling(14, min_periods=14).mean() / PIP
    x["range_vs_atr14"] = x.range_pips / x.atr14_pips.replace(0, np.nan)
    for n in [3, 20]:
        x[f"prior_{n}_high"] = x.high.shift(1).rolling(n, min_periods=n).max()
        x[f"prior_{n}_low"] = x.low.shift(1).rolling(n, min_periods=n).min()
    x["strong_bullish"] = (x.body_ratio >= MOM_BODY_RATIO) & (x.close_location >= 0.70)
    x["strong_bearish"] = (x.body_ratio >= MOM_BODY_RATIO) & (x.close_location <= 0.30)
    return x


def formal_events(seg: pd.DataFrame, direction: str, prior_direction: str, prior_start: float, prior_end: float) -> pd.DataFrame:
    x = features(seg)
    up = direction == "UP"

    x["breakout_A"] = (x.close > x.prior_20_high) if up else (x.close < x.prior_20_low)
    x["breakout_level"] = np.nan
    x.loc[x["breakout_A"], "breakout_level"] = x.loc[x["breakout_A"], "prior_20_high" if up else "prior_20_low"]
    last_level = np.nan
    age = 10_000
    vals = []
    for _, r in x.iterrows():
        if bool(r["breakout_A"]):
            last_level = float(r["breakout_level"])
            age = 0
        elif np.isfinite(last_level):
            age += 1
        vals.append((last_level, age))
    x["last_breakout_level"] = [v[0] for v in vals]
    x["bars_since_breakout"] = [v[1] for v in vals]
    tol = RETEST_ATR_MULT * x["atr14_pips"] * PIP
    if up:
        touched = x.low.le(x.last_breakout_level + tol) & x.high.ge(x.last_breakout_level - tol)
        rejected = x.close > x.open
    else:
        touched = x.high.ge(x.last_breakout_level - tol) & x.low.le(x.last_breakout_level + tol)
        rejected = x.close < x.open
    x["break_retest_B"] = touched & rejected & x.bars_since_breakout.between(1, RETEST_LOOKBACK)

    prior_range = abs(prior_end - prior_start)
    fib_mask = np.zeros(len(x), dtype=bool)
    fib_label = np.array([""] * len(x), dtype=object)
    if prior_range:
        for ratio in FIB_LEVELS:
            level = prior_end - prior_range * ratio / 100 if prior_direction == "UP" else prior_end + prior_range * ratio / 100
            dist = np.minimum(abs(x.high - level), abs(x.low - level)) / PIP
            near = dist <= FIB_ATR_TOL * x.atr14_pips
            if up:
                rej = (x.low <= level + FIB_ATR_TOL*x.atr14_pips*PIP) & (x.close > x.open) & (x.close > level)
            else:
                rej = (x.high >= level - FIB_ATR_TOL*x.atr14_pips*PIP) & (x.close < x.open) & (x.close < level)
            hit = near & rej
            new = hit & ~fib_mask
            fib_label[new] = str(ratio)
            fib_mask |= hit
    x["fib_rejection_C"] = fib_mask
    x["fib_rejection_level_C"] = fib_label

    if up:
        opposite = x.close < x.open
        structure_break = x.close > x.prior_3_high
    else:
        opposite = x.close > x.open
        structure_break = x.close < x.prior_3_low
    opp_recent = opposite.shift(1).rolling(STRUCTURE_WINDOW, min_periods=1).max().astype(bool)
    x["m1_structure_break_D"] = structure_break & opp_recent

    if up:
        x["momentum_E"] = x.strong_bullish & (x.range_vs_atr14 >= MOM_RANGE_ATR) & (x.return_3m_pips >= MOM_RETURN_3M)
    else:
        x["momentum_E"] = x.strong_bearish & (x.range_vs_atr14 >= MOM_RANGE_ATR) & (x.return_3m_pips <= -MOM_RETURN_3M)

    for name in ["breakout_A", "break_retest_B", "fib_rejection_C", "m1_structure_break_D", "momentum_E"]:
        x[name] = x[name].fillna(False).astype(bool)
    return x


def build() -> None:
    price = load_price()
    selected = load_selected().sort_values("swing_id")
    all_events, signal_rows, overlap_rows = [], [], []
    event_names = ["breakout_A", "break_retest_B", "fib_rejection_C", "m1_structure_break_D", "momentum_E"]

    for _, s in selected.iterrows():
        start, end = int(s.start_bar), int(s.end_bar)
        if start < 0 or end >= len(price) or start > end:
            raise ValueError(f"Invalid swing range {s.swing_id}: {start}-{end}")
        seg = formal_events(price.iloc[start:end+1], s.direction, s.prior_direction,
                            float(s.prior_start_price), float(s.prior_end_price))
        counts = {n: int(seg[n].sum()) for n in event_names}
        firsts = {}
        for n in event_names:
            idx = np.flatnonzero(seg[n].to_numpy())
            firsts[n] = int(idx[0]) if len(idx) else -1
            for i in idx:
                r = seg.iloc[i]
                all_events.append({
                    "swing_id": int(s.swing_id), "direction": s.direction, "event": n,
                    "bar_in_swing": int(i), "swing_position_pct": 100*i/max(1,len(seg)-1),
                    "timestamp": r.timestamp, "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                    "range_pips": r.range_pips, "body_ratio": r.body_ratio, "close_location": r.close_location,
                    "atr14_pips": r.atr14_pips, "range_vs_atr14": r.range_vs_atr14,
                    "return_3m_pips": r.return_3m_pips,
                    "fib_rejection_level_C": r.fib_rejection_level_C if n == "fib_rejection_C" else "",
                    "entry_reference": "NEXT_BAR_OPEN",
                })
        signal_rows.append({
            "swing_id": int(s.swing_id), "direction": s.direction, "swing_pips": float(s.swing_pips), "bars": len(seg),
            **{f"{n}_count": counts[n] for n in event_names},
            **{f"first_{n}_bar": firsts[n] for n in event_names},
            "any_formal_event": any(counts.values()),
            "distinct_models_present": sum(c > 0 for c in counts.values()),
        })
        for a in event_names:
            for b in event_names:
                if a < b:
                    overlap_rows.append({"swing_id": int(s.swing_id), "event_a": a, "event_b": b,
                                         "overlap_bars": int((seg[a] & seg[b]).sum())})

    events = pd.DataFrame(all_events)
    signals = pd.DataFrame(signal_rows)
    overlap = pd.DataFrame(overlap_rows)
    events.to_csv(OUT / "step14_formal_entry_events.csv", index=False)
    signals.to_csv(OUT / "step14_signal_event_study.csv", index=False)
    overlap.to_csv(OUT / "step14_event_overlap.csv", index=False)

    model_rows = []
    labels = {
        "breakout_A":"A_Breakout", "break_retest_B":"B_Break_Retest",
        "fib_rejection_C":"C_Fibonacci_Rejection", "m1_structure_break_D":"D_M1_Structure_Break",
        "momentum_E":"E_Momentum_Continuation"
    }
    for n, label in labels.items():
        c = int(signals[f"{n}_count"].gt(0).sum())
        bars = int(signals[f"{n}_count"].sum())
        first = signals.loc[signals[f"{n}_count"].gt(0), f"first_{n}_bar"]
        model_rows.append({"model":label,"event_column":n,"signals_with_event":c,"signal_pct":100*c/len(signals),
                           "total_event_bars":bars,"median_first_event_bar":first.median() if len(first) else np.nan})
    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(OUT / "step14_model_event_summary.csv", index=False)

    definitions = {
        "purpose":"Price Action Event Study + Formal Entry Definitions",
        "look_ahead_rule":"Event is known only at bar close; executable entry is NEXT_BAR_OPEN. No future bars are used to define the event.",
        "models": {
            "A_Breakout": "LONG: M1 close > prior 20-bar high. SHORT: M1 close < prior 20-bar low.",
            "B_Break_Retest": "After A, within 10 bars, price retests the broken level within 0.25 ATR14 and closes directionally (bullish for LONG, bearish for SHORT).",
            "C_Fibonacci_Rejection": "Price approaches/touches previous-swing 78.6/100/127.2/161.8/200% level within 0.25 ATR14 and closes away from the level in swing direction.",
            "D_M1_Structure_Break": "Within prior 10 bars there is an opposite-direction candle; current close breaks the prior 3-bar high (LONG) or low (SHORT).",
            "E_Momentum_Continuation": "Directional candle with body ratio >=0.60, range >=1.20 ATR14, and 3-minute return >=+0.50 pip (LONG) or <=-0.50 pip (SHORT).",
        },
        "parameters":{"break_lookback":20,"retest_lookback":10,"retest_atr":0.25,"fib_atr":0.25,"structure_lookback":3,"structure_window":10,"momentum_body_ratio":0.60,"momentum_range_atr":1.20,"momentum_return_3m_pips":0.50,"fib_levels":[78.6,100,127.2,161.8,200]},
        "next_step":"Step 15 backtests each model separately with entry at next-bar open; no parameter optimization in Step 14."
    }
    (OUT / "step14_entry_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    print("Step 14 — Price Action Event Study + Formal Entry Definitions")
    print(model_summary.to_string(index=False))
    print("Outputs:")
    for p in sorted(OUT.glob("*")):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    build()
