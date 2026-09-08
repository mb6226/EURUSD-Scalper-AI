from __future__ import annotations
import math, sys
from pathlib import Path
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from src.zigzag_mt5 import zigzag_mt5_with_point
M1_PATH=REPO_ROOT/'data/eurusd/EURUSD_1m.csv'; M5_PATH=REPO_ROOT/'data/eurusd/EURUSD_5m.csv'; RESULT_DIR=REPO_ROOT/'results/zigzag_step8'
POINT=0.00001; DEPTH=350; DEVIATION=125; BACKSTEP=40; LWMA_FAST=55; LWMA_SLOW=220

def load(path):
    df=pd.read_csv(path); lower={c.lower():c for c in df.columns}; tc=next(lower[x] for x in ('timestamp','datetime','time','date') if x in lower)
    cols={lower[x]:x for x in ('open','high','low','close') if x in lower}
    if len(cols)!=4: raise ValueError(f'Missing OHLC columns in {path}')
    df=df.rename(columns=cols); df.index=pd.to_datetime(df[tc],errors='coerce'); return df.loc[df.index.notna()].sort_index()[['open','high','low','close']]

def lwma(s, period):
    w=range(1,period+1); denom=period*(period+1)/2
    return s.rolling(period,min_periods=period).apply(lambda x: sum(v*ww for v,ww in zip(x,w))/denom,raw=True)

def main():
    m1=load(M1_PATH); m5=load(M5_PATH).copy(); m5['lwma55']=lwma(m5.close,LWMA_FAST); m5['lwma220']=lwma(m5.close,LWMA_SLOW)
    m5['trend']='NEUTRAL'; m5.loc[m5.lwma55>m5.lwma220,'trend']='UP'; m5.loc[m5.lwma55<m5.lwma220,'trend']='DOWN'
    zz=zigzag_mt5_with_point(m1.high.to_numpy(),m1.low.to_numpy(),depth=DEPTH,deviation_points=DEVIATION,backstep=BACKSTEP,point=POINT)
    piv=[]
    for i,v in enumerate(zz):
        if not math.isfinite(float(v)): continue
        v=float(v); kind='high' if abs(v-m1.high.iloc[i])<=abs(v-m1.low.iloc[i]) else 'low'; piv.append((i,m1.index[i],v,kind))
    swings=[]
    for n,(a,b) in enumerate(zip(piv[:-1],piv[1:]),1):
        ai,at,ap,ak=a; bi,bt,bp,bk=b; swings.append({'swing_id':n,'start_bar':ai,'end_bar':bi,'start_time':at,'end_time':bt,'start_kind':ak,'end_kind':bk,'start_price':ap,'end_price':bp,'direction':'UP' if bp>ap else 'DOWN','bars':bi-ai,'swing_pips':abs(bp-ap)*10000.0})
    rows=[]
    for s in swings:
        mid=(s['start_bar']+s['end_bar'])//2; mt=m1.index[mid]; pos=m5.index.searchsorted(mt,side='right')-1
        if pos<0: continue
        r=m5.iloc[pos]; tr=str(r.trend); aligned=(tr==s['direction']) if tr!='NEUTRAL' else False
        rows.append({**s,'mid_bar':mid,'mid_time':mt,'m5_time':m5.index[pos],'m5_close':float(r.close),'lwma55':float(r.lwma55) if pd.notna(r.lwma55) else math.nan,'lwma220':float(r.lwma220) if pd.notna(r.lwma220) else math.nan,'trend':tr,'swing_trend_aligned':aligned})
    out=pd.DataFrame(rows); RESULT_DIR.mkdir(parents=True,exist_ok=True); out.to_csv(RESULT_DIR/'swings_with_lwma_trend.csv',index=False)
    aligned=out[out.swing_trend_aligned==True].copy() if len(out) else out.copy(); aligned.to_csv(RESULT_DIR/'swings_lwma_trend_aligned.csv',index=False)
    summary=pd.DataFrame([{'depth':DEPTH,'deviation_points':DEVIATION,'backstep':BACKSTEP,'lwma_fast':LWMA_FAST,'lwma_slow':LWMA_SLOW,'pivot_count':len(piv),'swing_count':len(out),'trend_up_midpoints':int((out.trend=='UP').sum()),'trend_down_midpoints':int((out.trend=='DOWN').sum()),'trend_neutral_midpoints':int((out.trend=='NEUTRAL').sum()),'aligned_swing_count':len(aligned),'aligned_pct':len(aligned)/len(out)*100.0 if len(out) else math.nan,'up_swing_count':int((out.direction=='UP').sum()),'down_swing_count':int((out.direction=='DOWN').sum()),'aligned_up_count':int(((out.direction=='UP')&(out.swing_trend_aligned==True)).sum()),'aligned_down_count':int(((out.direction=='DOWN')&(out.swing_trend_aligned==True)).sum()),'median_aligned_swing_pips':aligned.swing_pips.median() if len(aligned) else math.nan,'mean_aligned_swing_pips':aligned.swing_pips.mean() if len(aligned) else math.nan,'median_aligned_bars':aligned.bars.median() if len(aligned) else math.nan,'mean_aligned_bars':aligned.bars.mean() if len(aligned) else math.nan}])
    summary.to_csv(RESULT_DIR/'lwma_trend_summary.csv',index=False)
    stats=[]
    for d in ('UP','DOWN'):
        x=out[out.direction==d]; a=x[x.swing_trend_aligned==True]; stats.append({'swing_direction':d,'swing_count':len(x),'aligned_count':len(a),'aligned_pct':len(a)/len(x)*100 if len(x) else math.nan,'median_swing_pips':x.swing_pips.median() if len(x) else math.nan,'median_aligned_swing_pips':a.swing_pips.median() if len(a) else math.nan,'median_bars':x.bars.median() if len(x) else math.nan,'median_aligned_bars':a.bars.median() if len(a) else math.nan})
    pd.DataFrame(stats).to_csv(RESULT_DIR/'lwma_direction_stats.csv',index=False)
    ts=[]
    for t in ('UP','DOWN','NEUTRAL'):
        x=out[out.trend==t]; ts.append({'midpoint_trend':t,'count':len(x),'pct':len(x)/len(out)*100 if len(out) else math.nan,'up_swings':int((x.direction=='UP').sum()),'down_swings':int((x.direction=='DOWN').sum()),'aligned_swings':int((x.swing_trend_aligned==True).sum())})
    pd.DataFrame(ts).to_csv(RESULT_DIR/'lwma_trend_state_stats.csv',index=False)
if __name__=='__main__': main()
