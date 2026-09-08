from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/step15_backtest"
OUT = ROOT / "results/step16_mfe_mae"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"A":"A_Breakout","B":"B_Break_Retest","C":"C_Fibonacci_Rejection","D":"D_M1_Structure_Break","E":"E_Momentum_Continuation"}
R_BINS = [-np.inf,-1.0,-0.5,0,0.5,1.0,1.5,2.0,3.0,np.inf]
R_LABELS = ["<-1R","-1 to -0.5R","-0.5 to 0R","0 to 0.5R","0.5 to 1R","1 to 1.5R","1.5 to 2R","2 to 3R",">=3R"]
MFE_BINS = [0,0.25,0.5,0.75,1,1.25,1.5,2,3,5,np.inf]
MFE_LABELS = ["0-0.25R","0.25-0.5R","0.5-0.75R","0.75-1R","1-1.25R","1.25-1.5R","1.5-2R","2-3R","3-5R",">=5R"]
MAE_BINS = [-np.inf,-2,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0]
MAE_LABELS = ["<=-2R","-2 to -1.5R","-1.5 to -1.25R","-1.25 to -1R","-1 to -0.75R","-0.75 to -0.5R","-0.5 to -0.25R","-0.25 to 0R"]

def load(model):
    p=IN/f"step15_trades_{model}.csv"
    t=pd.read_csv(p)
    req={"swing_id","direction","hold_bars","r_multiple","mfe_r","mae_r","exit_reason"}
    missing=req-set(t.columns)
    if missing: raise ValueError(f"{p} missing columns: {sorted(missing)}")
    for c in ["hold_bars","r_multiple","mfe_r","mae_r"]: t[c]=pd.to_numeric(t[c],errors="coerce")
    return t.dropna(subset=["r_multiple","mfe_r","mae_r"])

def distribution(t,col,bins,labels):
    x=pd.cut(t[col],bins=bins,labels=labels,right=False,include_lowest=True)
    out=x.value_counts(sort=False).rename_axis("bin").reset_index(name="count")
    out["pct"]=100*out["count"]/len(t)
    out["metric"]=col
    return out[["metric","bin","count","pct"]]

def main():
    all_trades={}; summaries=[]; dist_r=[]; dist_mfe=[]; dist_mae=[]; thresholds=[]; dirs=[]; wl=[]; overlap=[]
    for key,name in MODELS.items():
        t=load(key); all_trades[key]=t
        r=t.r_multiple
        summaries.append({"model":name,"trades":len(t),"mean_R":r.mean(),"median_R":r.median(),"std_R":r.std(ddof=1),"min_R":r.min(),"max_R":r.max(),"median_MFE_R":t.mfe_r.median(),"mean_MFE_R":t.mfe_r.mean(),"median_MAE_R":t.mae_r.median(),"mean_MAE_R":t.mae_r.mean(),"median_hold_bars":t.hold_bars.median(),"mfe_mae_abs_ratio":t.mfe_r.mean()/abs(t.mae_r.mean()) if t.mae_r.mean()!=0 else np.inf})
        dist_r.append(distribution(t,"r_multiple",R_BINS,R_LABELS).assign(model=name))
        dist_mfe.append(distribution(t,"mfe_r",MFE_BINS,MFE_LABELS).assign(model=name))
        dist_mae.append(distribution(t,"mae_r",MAE_BINS,MAE_LABELS).assign(model=name))
        for thr in [0.5,1.0,1.5,2.0]:
            thresholds.append({"model":name,"threshold_R":thr,"trades_reaching":int((t.mfe_r>=thr).sum()),"pct_reaching":100*(t.mfe_r>=thr).mean(),"trades_reaching_before_stop":int(((t.mfe_r>=thr)&(t.mae_r>-1)).sum()),"pct_reaching_before_stop":100*((t.mfe_r>=thr)&(t.mae_r>-1)).mean()})
        for d,g in t.groupby("direction"):
            dirs.append({"model":name,"direction":d,"trades":len(g),"win_rate_pct":100*(g.r_multiple>0).mean(),"mean_R":g.r_multiple.mean(),"median_R":g.r_multiple.median(),"mean_MFE_R":g.mfe_r.mean(),"median_MFE_R":g.mfe_r.median(),"mean_MAE_R":g.mae_r.mean(),"median_MAE_R":g.mae_r.median()})
        for outcome,g in t.assign(outcome=np.where(t.r_multiple>0,"winner",np.where(t.r_multiple<0,"loser","flat"))).groupby("outcome"):
            wl.append({"model":name,"outcome":outcome,"trades":len(g),"mean_MFE_R":g.mfe_r.mean(),"median_MFE_R":g.mfe_r.median(),"mean_MAE_R":g.mae_r.mean(),"median_MAE_R":g.mae_r.median(),"mean_R":g.r_multiple.mean()})
    for a in MODELS:
        for b in MODELS:
            if a>=b: continue
            sa=set(all_trades[a].swing_id.astype(int)); sb=set(all_trades[b].swing_id.astype(int)); inter=sa&sb; union=sa|sb
            overlap.append({"model_a":MODELS[a],"model_b":MODELS[b],"common_swing_ids":len(inter),"union_swing_ids":len(union),"jaccard_pct":100*len(inter)/len(union) if union else 0})
    pd.DataFrame(summaries).to_csv(OUT/"step16_model_distribution_summary.csv",index=False)
    pd.concat(dist_r,ignore_index=True).to_csv(OUT/"step16_r_distribution.csv",index=False)
    pd.concat(dist_mfe,ignore_index=True).to_csv(OUT/"step16_mfe_distribution.csv",index=False)
    pd.concat(dist_mae,ignore_index=True).to_csv(OUT/"step16_mae_distribution.csv",index=False)
    pd.DataFrame(thresholds).to_csv(OUT/"step16_threshold_reach.csv",index=False)
    pd.DataFrame(dirs).to_csv(OUT/"step16_direction_stats.csv",index=False)
    pd.DataFrame(wl).to_csv(OUT/"step16_winner_loser_excursion.csv",index=False)
    pd.DataFrame(overlap).to_csv(OUT/"step16_model_overlap.csv",index=False)
    config={"purpose":"Step 16 - MFE/MAE + R distribution diagnostics","source":"Step 15 trade-level CSVs","lookahead":"Outcome diagnostics only; no features are used for entry selection.","thresholds_R":[0.5,1.0,1.5,2.0],"before_stop_definition":"MFE threshold reached while MAE remains above -1R over the recorded trade path","note":"MFE/MAE are post-entry diagnostics and must not be used as predictive filters."}
    (OUT/"step16_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("Step 16 complete")
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=="__main__": main()
