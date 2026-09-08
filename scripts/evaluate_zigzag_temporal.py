from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

@njit(cache=True)
def roll_extreme(a, depth, is_high):
    n = a.size
    out = np.empty(n, np.float64)
    out[:] = np.nan
    q = np.empty(n, np.int64)
    head = 0
    tail = 0
    for i in range(n):
        while head < tail and q[head] <= i - depth:
            head += 1
        x = a[i]
        if is_high:
            while head < tail and a[q[tail - 1]] <= x:
                tail -= 1
        else:
            while head < tail and a[q[tail - 1]] >= x:
                tail -= 1
        q[tail] = i
        tail += 1
        if i >= depth - 1:
            out[i] = a[q[head]]
    return out

@njit(cache=True)
def pivots_mt5(high, low, depth, deviation, backstep, point):
    n = high.size
    start = depth
    dev = deviation * point
    hi = roll_extreme(high, depth, True)
    lo = roll_extreme(low, depth, False)
    hm = np.zeros(n, np.float64)
    lm = np.zeros(n, np.float64)
    last_lo = 0.0
    last_hi = 0.0
    for s in range(start, n):
        v = lo[s]
        if v == last_lo:
            v = 0.0
        else:
            last_lo = v
            if low[s] - v > dev:
                v = 0.0
            else:
                first = max(start, s - backstep)
                for j in range(s - 1, first - 1, -1):
                    if lm[j] != 0.0 and lm[j] > v:
                        lm[j] = 0.0
        if low[s] == v:
            lm[s] = v

        v = hi[s]
        if v == last_hi:
            v = 0.0
        else:
            last_hi = v
            if v - high[s] > dev:
                v = 0.0
            else:
                first = max(start, s - backstep)
                for j in range(s - 1, first - 1, -1):
                    if hm[j] != 0.0 and hm[j] < v:
                        hm[j] = 0.0
        if high[s] == v:
            hm[s] = v

    pi = np.empty(n, np.int64)
    pp = np.empty(n, np.float64)
    pc = 0
    mode = 0
    last_lo = 0.0
    last_hi = 0.0
    lp = -1
    hp = -1
    for s in range(start, n):
        if mode == 0:
            if last_lo == 0.0 and last_hi == 0.0:
                if hm[s] != 0.0:
                    last_hi = high[s]; hp = s; mode = -1
                    pi[pc] = s; pp[pc] = last_hi; pc += 1
                if lm[s] != 0.0:
                    last_lo = low[s]; lp = s; mode = 1
                    pi[pc] = s; pp[pc] = last_lo; pc += 1
        elif mode == 1:
            if lm[s] != 0.0 and lm[s] < last_lo and hm[s] == 0.0:
                if pc > 0 and lp == pi[pc - 1]: pc -= 1
                lp = s; last_lo = lm[s]
                pi[pc] = s; pp[pc] = last_lo; pc += 1
            if hm[s] != 0.0 and lm[s] == 0.0:
                hp = s; last_hi = hm[s]
                pi[pc] = s; pp[pc] = last_hi; pc += 1; mode = -1
        else:
            if hm[s] != 0.0 and hm[s] > last_hi and lm[s] == 0.0:
                if pc > 0 and hp == pi[pc - 1]: pc -= 1
                hp = s; last_hi = hm[s]
                pi[pc] = s; pp[pc] = last_hi; pc += 1
            if lm[s] != 0.0 and hm[s] == 0.0:
                lp = s; last_lo = lm[s]
                pi[pc] = s; pp[pc] = last_lo; pc += 1; mode = 1
    return pi[:pc], pp[:pc]


def metrics_for_period(d, start, end, depth, deviation, backstep, point):
    warmup = max(depth, backstep) + 5
    ws = max(0, start - warmup)
    x = d.iloc[ws:end].reset_index(drop=True)
    high = x.high.to_numpy(np.float64)
    low = x.low.to_numpy(np.float64)
    close = x.close.to_numpy(np.float64)
    pi, pp = pivots_mt5(high, low, int(depth), int(deviation), int(backstep), point)
    target = start - ws
    keep = pi >= target
    idx = pi[keep]
    prices = pp[keep]
    if idx.size < 2:
        return dict(pivot_count=0, swing_count=0, median_swing_pips=np.nan, median_bars=np.nan,
                    median_efficiency=np.nan, mean_efficiency=np.nan)
    # Include a predecessor pivot for the first target pivot when available.
    first = np.searchsorted(pi, target, side='left')
    if first > 0:
        idx = np.concatenate((pi[first-1:first], idx))
        prices = np.concatenate((pp[first-1:first], prices))
    swings = []
    bars = []
    eff = []
    for k in range(1, len(idx)):
        a, b = int(idx[k-1]), int(idx[k])
        if b < target:
            continue
        swing = abs(prices[k] - prices[k-1]) / (point * 10.0)
        path = np.abs(np.diff(close[a:b+1])).sum()
        efficiency = abs(close[b] - close[a]) / path if path > 0 else 0.0
        swings.append(swing); bars.append(b-a); eff.append(efficiency)
    if not swings:
        return dict(pivot_count=int(idx.size), swing_count=0, median_swing_pips=np.nan,
                    median_bars=np.nan, median_efficiency=np.nan, mean_efficiency=np.nan)
    return dict(
        pivot_count=int(idx.size), swing_count=len(swings),
        median_swing_pips=float(np.median(swings)), median_bars=float(np.median(bars)),
        median_efficiency=float(np.median(eff)), mean_efficiency=float(np.mean(eff)),
    )


def candidate_configs(path: Path) -> pd.DataFrame:
    top = pd.read_csv(path)
    cols = ['depth', 'deviation', 'backstep']
    top = top[cols].copy()
    return top.drop_duplicates().astype({'depth': int, 'deviation': int, 'backstep': int})


def periods(d, mode):
    ts = pd.to_datetime(d.timestamp)
    if mode == 'month':
        keys = ts.dt.to_period('M')
    else:
        keys = ts.dt.to_period('Q')
    out = []
    for key in keys.unique():
        mask = keys == key
        pos = np.flatnonzero(mask.to_numpy())
        if len(pos): out.append((str(key), int(pos[0]), int(pos[-1] + 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/eurusd/EURUSD_1m.parquet')
    ap.add_argument('--candidates', default='results/zigzag/top20.csv')
    ap.add_argument('--output-dir', default='results/zigzag_monthly')
    ap.add_argument('--mode', choices=['month', 'quarter'], required=True)
    ap.add_argument('--point', type=float, default=1e-5)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(args.input).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    cfg = candidate_configs(Path(args.candidates))
    rows = []
    for period, start, end in periods(d, args.mode):
        for r in cfg.itertuples(index=False):
            m = metrics_for_period(d, start, end, r.depth, r.deviation, r.backstep, args.point)
            rows.append(dict(period=period, depth=r.depth, deviation=r.deviation, backstep=r.backstep, **m))
    result = pd.DataFrame(rows)
    result['deviation_pips'] = result.deviation / 10.0
    result.to_csv(out / f'{args.mode}_all_results.csv', index=False)

    agg = result.groupby(['depth','deviation','backstep'], as_index=False).agg(
        periods=('period','count'),
        median_of_median_swing_pips=('median_swing_pips','median'),
        median_of_median_bars=('median_bars','median'),
        median_efficiency=('median_efficiency','median'),
        mean_efficiency=('mean_efficiency','mean'),
        mean_pivots=('pivot_count','mean'),
        min_pivots=('pivot_count','min'),
    )
    eff = result.pivot_table(index=['depth','deviation','backstep'], columns='period', values='median_efficiency')
    agg['efficiency_std'] = eff.std(axis=1).to_numpy()
    agg['efficiency_min'] = eff.min(axis=1).to_numpy()
    agg['efficiency_max'] = eff.max(axis=1).to_numpy()
    agg['efficiency_stability'] = agg.efficiency_min / agg.efficiency_max.replace(0, np.nan)
    agg = agg.sort_values(['efficiency_stability','median_efficiency','mean_efficiency'], ascending=False)
    agg.to_csv(out / f'{args.mode}_stability.csv', index=False)
    agg.head(20).to_csv(out / f'{args.mode}_top20.csv', index=False)
    (out / 'README.md').write_text(
        f'# EURUSD M1 ZigZag {args.mode.title()} Stability\n\n'
        f'Candidates are inherited from the previous optimization step: {args.candidates}.\n'
        f'Each period is evaluated with historical warmup so ZigZag state is not reset at the boundary.\n'
        f'Outputs contain per-period results plus cross-period stability rankings.\n', encoding='utf-8')
    print(agg.head(20).to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
