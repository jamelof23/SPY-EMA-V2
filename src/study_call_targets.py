"""Exploratory 2020–2024 target-before-stop sensitivity for the chart example."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate(daily,bars,pairs):
    # Leave the 2025 validation observations out of this new design study.
    pairs=pairs.loc[pairs.index.year<=2024]
    ds,bs=pairs.daily_end.to_numpy(),pairs.intraday_end.to_numpy()
    context=daily.iloc[ds]
    fast=bars.ema8.to_numpy();slow=bars.ema21.to_numpy()
    cross=(context.close.to_numpy()>context.open.to_numpy()) & (fast[bs]>slow[bs]) & (fast[bs-1]<=slow[bs-1])
    cross &= bars.index[bs]-bars.index[bs-1]==pd.Timedelta(minutes=5)
    triggered=pairs.loc[cross]
    # The last observed close is the actual session end for every included day in the prepared data.
    by_day=pd.Series(pairs.index,index=pairs.index).groupby(pairs.index.normalize()).max()
    assert by_day.map(lambda t: (t.hour,t.minute) in ((16,0),(13,0))).all()
    history=[]
    for label in ('ema8','ema21'):
        for min_rr in (0.,1.):
            chosen=set()
            for row in triggered.itertuples():
                t=row.Index;day=t.normalize()
                if day in chosen or t>=by_day[day] or t not in bars.index:continue
                old=daily.iloc[row.daily_end];stop=old.close;target=old[label]
                signal=bars.close.iloc[row.intraday_end];entry=bars.at[t,'open']
                if not (stop<signal<target and stop<entry<target):continue
                rr=(target-entry)/(entry-stop)
                if rr<min_rr:continue
                chosen.add(day)
                status='close';price=np.nan;when=pd.NaT
                for at in pd.date_range(t,by_day[day],freq='5min',inclusive='left'):
                    if at not in bars.index or bars.at[at,'minute_count']<5:status='incomplete';break
                    a=bars.loc[at]
                    if a.open<=stop:status='stop';price=a.open
                    elif a.open>=target:status='target';price=target
                    elif a.low<=stop and a.high>=target:status='ambiguous'
                    elif a.low<=stop:status='stop';price=stop
                    elif a.high>=target:status='target';price=target
                    if status!='close':
                        when=at;break
                    price=a.close;when=at
                realized=(price-entry)/entry*10000 if status not in ('incomplete','ambiguous') else np.nan
                history.append((t,label,min_rr,signal,entry,stop,target,rr,status,when,realized))
    led=pd.DataFrame(history,columns=['decision_et','target_ema','min_planned_reward_risk','signal_close','entry_open','stop','target','planned_reward_risk','outcome','exit_bar_open_et','underlying_return_bp'])
    led['split']=np.where(led.decision_et.dt.year<=2023,'fit_2020_23','dev_2024')
    counts=led.groupby(['target_ema','min_planned_reward_risk','split','outcome']).size().unstack(fill_value=0)
    g=led.groupby(['target_ema','min_planned_reward_risk','split'])
    stats=g.agg(n=('outcome','size'),median_planned_reward_risk=('planned_reward_risk','median'),mean_spy_bp=('underlying_return_bp','mean'),median_spy_bp=('underlying_return_bp','median')).join(counts)
    return led,stats


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data');a.add_argument('output');args=a.parse_args()
    p=Path(args.data);out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    frames=[pd.read_parquet(p/(f+'.parquet')) for f in ('daily','five_minute','pairs')]
    ledger,summary=evaluate(*frames);ledger.to_csv(out/'development_trades.csv',index=False);summary.to_csv(out/'development_summary.csv');print(summary.to_string())
