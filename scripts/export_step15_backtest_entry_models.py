from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE = ROOT / "data/eurusd/EURUSD_1m.csv"
EVENTS = ROOT / "results/step14_price_action_event_study/step14_formal_entry_events.csv"
OUT = ROOT / "results/step15_backtest"
OUT.mkdir(parents=True, exist_ok=True)
PIP = 0.0001
SL_ATR = 1.0
TP_ATR = 1.5
MAX_HOLD_BARS = 120
MODELS = {"A_Breakout":"breakout_A","B_Break_Retest":"break_retest_B","C_Fibonacci_Rejection":"fib_rejection_C","D_M1_Structure_Break":"m1_structure_break_D","E_Momentum_Continuation":"momentum_E"}

def load_price():
    x=pd.read_csv(PRICE); req={"timestamp","open","high","low","close"}; missing=req-set(x.columns)
    if missing: raise ValueError(f"Price data missing columns: {sorted(missing)}")
    x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce")
    for c in ["open","high","low","close"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").reset_index(drop=True)

def load_events():
    x=pd.read_csv(EVENTS); req={"swing_id","direction","event","bar_in_swing","timestamp","atr14_pips"}; missing=req-set(x.columns)
    if missing: raise ValueError(f"Step 14 event data missing columns: {sorted(missing)}")
    x["timestamp"]=pd.to_datetime(x["timestamp"],errors="coerce"); x["bar_in_swing"]=pd.to_numeric(x["bar_in_swing"],errors="coerce"); x["atr14_pips"]=pd.to_numeric(x["atr14_pips"],errors="coerce")
    return x.dropna(subset=["timestamp","bar_in_swing","atr14_pips"]).copy()

def backtest_model(price, events, model, event_col):
    ev=events[events.event.eq(event_col)].copy().sort_values(["swing_id","bar_in_swing","timestamp"]).drop_duplicates("swing_id",keep="first")
    rows=[]
    for _,e in ev.iterrows():
        candidates=price.index[price.timestamp>e.timestamp]
        if len(candidates)==0: continue
        entry_i=int(candidates[0]); atr=float(e.atr14_pips)*PIP
        if not np.isfinite(atr) or atr<=0: continue
        side=1 if str(e.direction)=="UP" else -1; entry=float(price.loc[entry_i,"open"]); risk=atr*SL_ATR
        target=entry+side*atr*TP_ATR; stop=entry-side*risk; last_i=min(len(price)-1,entry_i+MAX_HOLD_BARS)
        exit_i=last_i; reason="TIMEOUT"; exit_price=float(price.loc[last_i,"close"])
        for j in range(entry_i,last_i+1):
            hi=float(price.loc[j,"high"]); lo=float(price.loc[j,"low"])
            hit_sl=(lo<=stop) if side==1 else (hi>=stop); hit_tp=(hi>=target) if side==1 else (lo<=target)
            if hit_sl and hit_tp: exit_i,reason,exit_price=j,"SL_AND_TP_SAME_BAR_SL_FIRST",stop; break
            if hit_sl: exit_i,reason,exit_price=j,"SL",stop; break
            if hit_tp: exit_i,reason,exit_price=j,"TP",target; break
        if reason=="TIMEOUT": exit_price=float(price.loc[exit_i,"close"])
        r=side*(exit_price-entry)/risk; future=price.iloc[entry_i:exit_i+1]
        if side==1: mfe=(float(future.high.max())-entry)/risk; mae=(float(future.low.min())-entry)/risk
        else: mfe=(entry-float(future.low.min()))/risk; mae=(entry-float(future.high.max()))/risk
        rows.append({"model":model,"event":event_col,"swing_id":int(e.swing_id),"direction":e.direction,"event_time":e.timestamp,"entry_time":price.loc[entry_i,"timestamp"],"entry_bar_index":entry_i,"entry_price":entry,"atr14_pips":float(e.atr14_pips),"sl_price":stop,"tp_price":target,"exit_time":price.loc[exit_i,"timestamp"],"exit_price":exit_price,"exit_reason":reason,"hold_bars":exit_i-entry_i,"r_multiple":r,"mfe_r":mfe,"mae_r":mae})
    return pd.DataFrame(rows)

def summarize(t,model,event):
    if t.empty: return {"model":model,"event":event,"trades":0}
    r=t.r_multiple; wins=r>0; losses=r<0; gp=r[r>0].sum(); gl=-r[r<0].sum(); eq=r.cumsum(); dd=eq-eq.cummax()
    return {"model":model,"event":event,"trades":len(r),"wins":int(wins.sum()),"losses":int(losses.sum()),"timeouts":int(t.exit_reason.eq("TIMEOUT").sum()),"win_rate_pct":100*wins.mean(),"avg_R":r.mean(),"median_R":r.median(),"std_R":r.std(ddof=1) if len(r)>1 else 0.0,"expectancy_R":r.mean(),"profit_factor":gp/gl if gl>0 else np.inf,"total_R":r.sum(),"max_drawdown_R":dd.min(),"median_MFE_R":t.mfe_r.median(),"median_MAE_R":t.mae_r.median(),"mean_MFE_R":t.mfe_r.mean(),"mean_MAE_R":t.mae_r.mean(),"median_hold_bars":t.hold_bars.median()}

def main():
    price=load_price(); events=load_events(); summaries=[]
    config={"purpose":"Step 15 - Backtest Entry Models A-E","entry_rule":"Event known only at bar close; entry is NEXT_BAR_OPEN.","trade_selection":"First valid event per swing per model.","sl":"1.0 x ATR14 at event bar, fixed at entry.","tp":"1.5 x ATR14 at event bar, fixed at entry.","max_hold_bars":MAX_HOLD_BARS,"intrabar_ambiguity":"If SL and TP are both touched in the same M1 candle, SL is assumed first (conservative).","optimization":False,"costs":"Excluded; Step 17."}
    for model,event in MODELS.items():
        t=backtest_model(price,events,model,event); t.to_csv(OUT/f"step15_trades_{model[0]}.csv",index=False); summaries.append(summarize(t,model,event))
    s=pd.DataFrame(summaries); s.to_csv(OUT/"step15_model_summary.csv",index=False); (OUT/"step15_backtest_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("Step 15 - Backtest Entry Models A-E"); print(s.to_string(index=False))

if __name__=="__main__": main()
