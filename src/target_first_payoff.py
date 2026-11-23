"""Evaluate predeclared stock-price filters on out-of-year barrier forecasts."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def outcomes(bars,pairs,events):
    last=pd.Series(pairs.index,index=pairs.index).groupby(pairs.index.normalize()).max()
    rows=[]
    for r in events.itertuples():
        t=r.Index; end=last[t.normalize()];entry=bars.at[t,'open'];price=np.nan
        for at in pd.date_range(t,end,freq='5min',inclusive='left'):
            a=bars.loc[at]
            if a.open>=r.target: price=r.target;break
            if a.open<=r.stop: price=a.open;break
            if a.high>=r.target and a.low<=r.stop: raise ValueError('unexpected unresolved same-bar touch')
            if a.high>=r.target:price=r.target;break
            if a.low<=r.stop:price=r.stop;break
            price=a.close
        rows.append((entry,price,10000*(price/entry-1)))
    return pd.DataFrame(rows,index=events.index,columns=['entry','exit','gross_spy_bp'])


def run(bars,pairs,events,preds,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    events=events.reindex(preds.index).copy();assert events.index.max().year==2024
    x=events.join(outcomes(bars,pairs,events))
    x['probability']=preds.distance_linear
    x['planned_reward_risk']=(x.target-x.signal)/(x.signal-x.stop)
    x['conservative_expected_bp']=(x.probability*(x.target-x.signal)-(1-x.probability)*(x.signal-x.stop))/x.signal*10000
    # These are exploratory fixed filters. Score from the completed bar. Only one entry per day.
    rules={'first_eligible':pd.Series(True,index=x.index),'probability_60pct':x.probability>=.60,
           'expected_above_2bp':x.conservative_expected_bp>2,'reward_risk_at_least_one':x.planned_reward_risk>=1}
    days=x.index.normalize().unique();records=[];daily=pd.DataFrame(index=days)
    for name,rule in rules.items():
        sub=x.loc[rule]; one=sub.loc[~sub.index.normalize().duplicated()]
        one=one.copy();one['session']=one.index.normalize()
        value=(one.set_index('session').gross_spy_bp-2).reindex(days,fill_value=0)
        daily[name]=value
        for year in (2022,2023,2024,'all'):
            mask=np.ones(len(one),dtype=bool) if year=='all' else one.index.year==year
            trade=one[mask]
            part=value if year=='all' else value[value.index.year==year]
            rng=np.random.default_rng(81);v=part.to_numpy();boot=v[rng.integers(len(v),size=(2500,len(v)))].mean(1)
            records.append(dict(rule=name,year=year,opportunity_days=len(part),entries=len(trade),
                                 target_first=int((trade.result=='target').sum()),stop_first=int((trade.result=='stop').sum()),
                                 close=int((trade.result=='neither').sum()),
                                 mean_gross_bp_per_position=trade.gross_spy_bp.mean(),mean_net_bp_per_opportunity_day=v.mean(),
                                 day_bootstrap_low=np.quantile(boot,.025),day_bootstrap_high=np.quantile(boot,.975)))
    ans=pd.DataFrame(records);ans.to_csv(out/'payoff_filters.csv',index=False);daily.to_csv(out/'daily_net_spy_bp.csv')
    print(ans[ans.year=='all'].to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data');a.add_argument('events');a.add_argument('predictions');a.add_argument('out');s=a.parse_args();p=Path(s.data)
    run(pd.read_parquet(p/'five_minute.parquet'),pd.read_parquet(p/'pairs.parquet'),pd.read_csv(s.events,index_col='decision_et',parse_dates=True),pd.read_csv(s.predictions,index_col='decision_et',parse_dates=True),s.out)
