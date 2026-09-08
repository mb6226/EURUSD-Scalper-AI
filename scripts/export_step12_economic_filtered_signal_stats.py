from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/step11_economic_regime/step11_filtered_signals.csv"
OUT = ROOT / "results/step12_economic_filtered_stats"
OUT.mkdir(parents=True, exist_ok=True)
FIB = ["<0","0-23.6","23.6-38.2","38.2-50","50-61.8","61.8-78.6","78.6-100","100-127.2","127.2-161.8","161.8-200","200-261.8","261.8-361.8","361.8-423.6",">=423.6"]


def stat_rows(df, groups, label):
    metrics = ["swing_pips","prior_swing_pips","bars","prior_bars","reversal_pct","current_end_retracement_pct","confidence"]
    rows=[]
    grouped=[((),df)] if not groups else df.groupby(groups, dropna=False, sort=True)
    for key,g in grouped:
        key=key if isinstance(key,tuple) else (key,)
        row={c:v for c,v in zip(groups,key)}; row["scope"]=label; row["count"]=len(g)
        for m in metrics:
            s=pd.to_numeric(g[m],errors="coerce").dropna()
            row[m+"_mean"]=s.mean(); row[m+"_median"]=s.median(); row[m+"_std"]=s.std(ddof=1); row[m+"_p25"]=s.quantile(.25); row[m+"_p75"]=s.quantile(.75); row[m+"_min"]=s.min(); row[m+"_max"]=s.max()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    if not INPUT.exists(): raise FileNotFoundError(INPUT)
    df=pd.read_csv(INPUT)
    required={"swing_id","direction","swing_pips","bars","prior_direction","prior_swing_pips","prior_bars","reversal_pct","reversal_fib_zone","current_end_retracement_pct","current_end_fib_zone","signal_date","month","economic_regime","confidence"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df)!=162: raise ValueError(f"Expected 162 signals, found {len(df)}")
    if df.swing_id.duplicated().any(): raise ValueError("Duplicate swing_id")
    numeric=["swing_pips","prior_swing_pips","bars","prior_bars","reversal_pct","current_end_retracement_pct","confidence"]
    for c in numeric: df[c]=pd.to_numeric(df[c],errors="coerce")
    if df[numeric].isna().any().any(): raise ValueError("Invalid numeric values")

    df.sort_values(["signal_date","swing_id"]).assign(step12_signal_rank=range(1,163)).to_csv(OUT/"step12_162_signal_list.csv",index=False)
    stat_rows(df,[],"overall").to_csv(OUT/"step12_overall_stats.csv",index=False)
    stat_rows(df,["direction"],"direction").to_csv(OUT/"step12_direction_stats.csv",index=False)
    fs=stat_rows(df,["reversal_fib_zone"],"fibonacci_zone")
    counts=df.reversal_fib_zone.value_counts().reindex(FIB,fill_value=0)
    fs=fs.set_index("reversal_fib_zone").reindex(FIB).reset_index(); fs["count"]=counts.values; fs["pct_of_162"]=counts.values/162*100
    fs.to_csv(OUT/"step12_fibonacci_zone_stats.csv",index=False)

    bins=[-float("inf"),10,20,30,40,50,75,100,150,200,300,float("inf")]
    labels=["<10","10-20","20-30","30-40","40-50","50-75","75-100","100-150","150-200","200-300",">=300"]
    sb=pd.cut(df.swing_pips,bins=bins,labels=labels,right=False).value_counts().reindex(labels,fill_value=0)
    pd.DataFrame({"swing_pips_bin":labels,"count":sb.values,"pct_of_162":sb.values/162*100}).to_csv(OUT/"step12_swing_pips_distribution.csv",index=False)
    stat_rows(df,["month"],"month").to_csv(OUT/"step12_monthly_stats.csv",index=False)
    m=df.groupby(["economic_regime","direction"],dropna=False).size().rename("count").reset_index(); m["pct_of_162"]=m["count"]/162*100; m.to_csv(OUT/"step12_economic_direction_matrix.csv",index=False)

    r=df.reversal_pct; s=df.swing_pips
    print("Step 12 — Economic-filtered 162 signal statistical analysis")
    print(f"Signals: {len(df)}")
    print(f"Swing pips: median={s.median():.2f}, mean={s.mean():.2f}, p25={s.quantile(.25):.2f}, p75={s.quantile(.75):.2f}, min={s.min():.2f}, max={s.max():.2f}")
    print(f"Prior swing pips: median={df.prior_swing_pips.median():.2f}, mean={df.prior_swing_pips.mean():.2f}")
    print(f"Bars: median={df.bars.median():.0f}, mean={df.bars.mean():.2f}")
    print(f"Fibo reversal: median={r.median():.2f}%, mean={r.mean():.2f}%, p25={r.quantile(.25):.2f}%, p75={r.quantile(.75):.2f}%, min={r.min():.2f}%, max={r.max():.2f}%")
    print(f">=100%: {(r>=100).sum()} ({(r>=100).mean()*100:.2f}%) | >=127.2%: {(r>=127.2).sum()} ({(r>=127.2).mean()*100:.2f}%) | >=161.8%: {(r>=161.8).sum()} ({(r>=161.8).mean()*100:.2f}%) | >=200%: {(r>=200).sum()} ({(r>=200).mean()*100:.2f}%)")

if __name__=="__main__": main()
