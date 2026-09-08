from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

DEPTHS = np.arange(50, 501, 50, dtype=np.int64)
DEVIATIONS = np.arange(25, 251, 25, dtype=np.int64)
BACKSTEPS = np.arange(20, 201, 20, dtype=np.int64)

@njit(cache=True)
def roll_extreme(a, depth, is_high):
    n = a.size; out = np.empty(n, np.float64); out[:] = np.nan
    q = np.empty(depth + 2, np.int64); head = 0; tail = 0
    for i in range(n):
        while head < tail and q[head] <= i - depth: head += 1
        x = a[i]
        if is_high:
            while head < tail and a[q[tail-1]] <= x: tail -= 1
        else:
            while head < tail and a[q[tail-1]] >= x: tail -= 1
        q[tail] = i; tail += 1
        if i >= depth - 1: out[i] = a[q[head]]
        if head > depth:
            j = 0
            for k in range(head, tail): q[j] = q[k]; j += 1
            tail = j; head = 0
    return out

@njit(cache=True)
def evaluate_one(high, low, close, depth, deviation, backstep, point):
    n = high.size; start = depth - 1; dev = deviation * point
    hi = roll_extreme(high, depth, True); lo = roll_extreme(low, depth, False)
    hm = np.zeros(n, np.float64); lm = np.zeros(n, np.float64)
    last_lo = 0.0; last_hi = 0.0
    for s in range(start, n):
        v = lo[s]
        if v == last_lo: v = 0.0
        else:
            last_lo = v
            if low[s] - v > dev: v = 0.0
            else:
                for j in range(s-1, max(start, s-backstep)-1, -1):
                    if lm[j] != 0.0 and lm[j] > v: lm[j] = 0.0
        if low[s] == v: lm[s] = v
        v = hi[s]
        if v == last_hi: v = 0.0
        else:
            last_hi = v
            if v - high[s] > dev: v = 0.0
            else:
                for j in range(s-1, max(start, s-backstep)-1, -1):
                    if hm[j] != 0.0 and hm[j] < v: hm[j] = 0.0
        if high[s] == v: hm[s] = v

    pi = np.empty(n, np.int64); pp = np.empty(n, np.float64); pc = 0
    mode = 0; last_lo = 0.0; last_hi = 0.0; lp = -1; hp = -1
    for s in range(start, n):
        if mode == 0:
            if last_lo == 0.0 and last_hi == 0.0:
                if hm[s] != 0.0:
                    last_hi = high[s]; hp = s; mode = -1; pi[pc] = s; pp[pc] = last_hi; pc += 1
                if lm[s] != 0.0:
                    last_lo = low[s]; lp = s; mode = 1; pi[pc] = s; pp[pc] = last_lo; pc += 1
        elif mode == 1:
            if lm[s] != 0.0 and lm[s] < last_lo and hm[s] == 0.0:
                if pc > 0 and lp == pi[pc-1]: pc -= 1
                lp = s; last_lo = lm[s]; pi[pc] = s; pp[pc] = last_lo; pc += 1
            if hm[s] != 0.0 and lm[s] == 0.0:
                hp = s; last_hi = hm[s]; pi[pc] = s; pp[pc] = last_hi; pc += 1; mode = -1
        else:
            if hm[s] != 0.0 and hm[s] > last_hi and lm[s] == 0.0:
                if pc > 0 and hp == pi[pc-1]: pc -= 1
                hp = s; last_hi = hm[s]; pi[pc] = s; pp[pc] = last_hi; pc += 1
            if lm[s] != 0.0 and hm[s] == 0.0:
                lp = s; last_lo = lm[s]; pi[pc] = s; pp[pc] = last_lo; pc += 1; mode = 1
    if pc < 4: return 0.0, 0.0, 0.0, 0.0, 0.0

    sw = np.empty(pc-1, np.float64); bars = np.empty(pc-1, np.float64); ef = np.empty(pc-1, np.float64)
    for k in range(pc-1):
        a = pi[k]; b = pi[k+1]; sw[k] = abs(pp[k+1]-pp[k])/(point*10.0); bars[k] = b-a
        path = 0.0
        for j in range(a+1, b+1): path += abs(close[j]-close[j-1])
        ef[k] = abs(close[b]-close[a])/path if path > 0.0 else 0.0
    sw.sort(); bars.sort(); ef.sort(); m = sw.size//2
    med_sw = sw[m] if sw.size%2 else 0.5*(sw[m-1]+sw[m])
    med_b = bars[m] if bars.size%2 else 0.5*(bars[m-1]+bars[m])
    med_ef = ef[m] if ef.size%2 else 0.5*(ef[m-1]+ef[m])
    return float(pc), med_sw, med_b, med_ef, np.mean(ef)

@njit(parallel=True, cache=True)
def sweep(high, low, close, depths, deviations, backsteps, point):
    m = depths.size * deviations.size * backsteps.size; out = np.empty((m, 8), np.float64)
    for flat in prange(m):
        a = flat // (deviations.size*backsteps.size); r = flat % (deviations.size*backsteps.size)
        b = r // backsteps.size; c = r % backsteps.size
        pc, ms, mb, me, ae = evaluate_one(high, low, close, int(depths[a]), int(deviations[b]), int(backsteps[c]), point)
        out[flat,0]=depths[a]; out[flat,1]=deviations[b]; out[flat,2]=backsteps[c]; out[flat,3]=pc; out[flat,4]=ms; out[flat,5]=mb; out[flat,6]=me; out[flat,7]=ae
    return out

def rank(df):
    density = df.pivot_count.clip(lower=1)
    density_score = np.exp(-((np.log(density)-np.log(500.0))**2)/(2*1.35**2))
    swing_score = np.clip(np.log1p(df.median_swing_pips)/np.log1p(25.0), 0, 1)
    eff_score = np.clip(df.median_efficiency, 0, 1)
    spacing = np.exp(-((np.log1p(df.median_bars)-np.log1p(25.0))**2)/(2*1.25**2))
    df = df.copy(); df['score'] = 100*(.35*eff_score + .30*swing_score + .20*density_score + .15*spacing)
    return df.sort_values('score', ascending=False).reset_index(drop=True)

def stable_regions(df):
    rows=[]
    for _, r in df.head(50).iterrows():
        mask=(abs(df.depth-r.depth)<=50)&(abs(df.deviation-r.deviation)<=25)&(abs(df.backstep-r.backstep)<=20)
        s=df.loc[mask,'score']
        if len(s)>=5:
            rows.append(dict(depth=int(r.depth), deviation=int(r.deviation), backstep=int(r.backstep), score=float(r.score), neighborhood_n=len(s), neighborhood_mean=float(s.mean()), neighborhood_median=float(s.median()), neighborhood_min=float(s.min()), neighborhood_max=float(s.max()), stability=float(s.mean()/max(r.score,1e-9))))
    return pd.DataFrame(rows).sort_values(['stability','neighborhood_mean','score'], ascending=False).drop_duplicates(['depth','deviation','backstep']).reset_index(drop=True) if rows else pd.DataFrame()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='data/eurusd/EURUSD_1m.parquet'); ap.add_argument('--output-dir',default='results/zigzag'); ap.add_argument('--point',type=float,default=1e-5); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(args.input).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    h,l,c=(d[x].to_numpy(np.float64) for x in ['high','low','close'])
    raw=sweep(h,l,c,DEPTHS,DEVIATIONS,BACKSTEPS,args.point)
    r=pd.DataFrame(raw,columns=['depth','deviation','backstep','pivot_count','median_swing_pips','median_bars','median_efficiency','mean_efficiency'])
    r=rank(r); r['deviation_pips']=r.deviation/10.0
    s=stable_regions(r)
    if not s.empty: s['deviation_pips']=s.deviation/10.0
    r.to_csv(out/'all_results.csv',index=False); r.head(20).to_csv(out/'top20.csv',index=False); s.head(20).to_csv(out/'stable_regions.csv',index=False); r.head(20).to_json(out/'top20.json',orient='records',indent=2)
    b=r.iloc[0]
    (out/'README.md').write_text(f'''# EURUSD M1 MT5 ZigZag Optimization\n\nRows: {len(d):,}\nConfigurations: 1,000\nPoint: {args.point:g}\nDeviation is MT5 points; on 5-digit EURUSD, 10 points = 1 pip.\n\n## Best cell\nDepth={int(b.depth)}, Deviation={int(b.deviation)} points ({b.deviation_pips:.1f} pips), Backstep={int(b.backstep)}, Score={b.score:.3f}.\n\nStable regions rank nearby parameter neighborhoods; prefer a robust region over an isolated peak. This is a research ranking, not a future-performance guarantee.\n''',encoding='utf-8')
    print(r.head(20).to_string(index=False)); print('\nSTABLE REGIONS\n'); print(s.head(20).to_string(index=False))

if __name__=='__main__': main()
