"""Development-only study of reaching the prior daily EMA8 before the prior close.

Includes fixed clock observations with a usable upside target and downside stop.
The outputs are SPY barrier events. No-trade intent and options P/L are unobserved.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from paired_baselines import features


def barriers(daily,bars,pairs):
    pairs=pairs.loc[pairs.index.year<=2024]
    last_by_day=pd.Series(pairs.index,index=pairs.index).groupby(pairs.index.normalize()).max()
    selected=pairs.loc[pairs.index.minute.isin([5,35])]
    data=[]
    for item in selected.itertuples():
        t=item.Index;end=last_by_day.at[t.normalize()]
        if t>=end or t not in bars.index:continue
        old=daily.iloc[item.daily_end]
        stop,target=old.close,old.ema8
        mid=bars.close.iloc[item.intraday_end];entry=bars.at[t,'open']
        if not(stop<mid<target and stop<entry<target):continue
        result='neither'
        for tick in pd.date_range(t,end,freq='5min',inclusive='left'):
            if tick not in bars.index or bars.at[tick,'minute_count']!=5:
                result='unknown';break
            bar=bars.loc[tick]
            if bar.open>=target:result='target'
            elif bar.open<=stop:result='stop'
            elif bar.high>=target and bar.low<=stop:result='unknown'
            elif bar.high>=target:result='target'
            elif bar.low<=stop:result='stop'
            if result!='neither':break
        if result!='unknown':data.append((t,result,target,stop,mid))
    return pd.DataFrame(data,columns=['decision_et','result','target','stop','signal']).set_index('decision_et')


def run(daily,bars,pairs,dest):
    dest=Path(dest);dest.mkdir(exist_ok=True,parents=True)
    events=barriers(daily,bars,pairs)
    sub=pairs.loc[events.index]
    clock,_,ctx=features(daily,bars,sub)
    closed=events.signal
    scale=(bars.high-bars.low).rolling(12).mean().to_numpy()[sub.intraday_end].clip(.02)
    simple=clock.copy()
    simple['room_to_target']=(events.target-closed)/scale
    simple['room_to_stop']=(closed-events.stop)/scale
    x=simple.join(ctx.drop(columns=clock.columns))
    y=(events.result=='target').astype(int)
    train=y.index.year<=2023;test=y.index.year==2024
    # Give every represented session equal weight at fitting and evaluation.
    weights=pd.Series(1.,index=y.index)/y.groupby(y.index.normalize()).transform('size')
    weights/=weights[train].mean()
    prediction=pd.DataFrame({'result':events.result[test],'target_first':y[test]})
    result={'train_rows':int(train.sum()),'train_sessions':y[train].index.normalize().nunique(),'dev_rows':int(test.sum()),
            'dev_sessions':y[test].index.normalize().nunique(),'dev_event_counts':events.result[test].value_counts().to_dict(),'models':{}}
    prior=np.average(y[train],weights=weights[train])
    prediction['prior']=prior
    reg=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=300))
    reg.fit(simple[train],y[train],logisticregression__sample_weight=weights[train]);prediction['distance_time']=reg.predict_proba(simple[test])[:,1]
    tree=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=3,early_stopping=False,random_state=41)
    tree.fit(x[train],y[train],sample_weight=weights[train]);prediction['chart_and_levels']=tree.predict_proba(x[test])[:,1]
    for name in ('prior','distance_time','chart_and_levels'):
        p=prediction[name];truth=y[test];w=weights[test]
        result['models'][name]={'day_weighted_auc':float(roc_auc_score(truth,p,sample_weight=w)),
          'day_weighted_balanced_accuracy':float(balanced_accuracy_score(truth,p>=.5,sample_weight=w)),
          'day_weighted_log_loss':float(log_loss(truth,p,sample_weight=w,labels=[0,1]))}
    # Test how confidently any differences transfer across trading sessions.
    truth=prediction.target_first.to_numpy()
    a=prediction.chart_and_levels.to_numpy();b=prediction.distance_time.to_numpy()
    delta=np.log(np.where(truth,b,1-b))-np.log(np.where(truth,a,1-a))
    session=pd.Series(delta,index=prediction.index).groupby(prediction.index.normalize()).mean().to_numpy()
    rng=np.random.default_rng(121);boot=session[rng.integers(len(session),size=(2500,len(session)))].mean(1)
    result['context_minus_distance_log_loss']={'mean':float(session.mean()),'session_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
    joblib.dump(reg,dest/'distance_time.joblib');joblib.dump(tree,dest/'chart_and_levels.joblib')
    prediction.to_csv(dest/'dev_2024_predictions.csv');events.to_csv(dest/'development_barrier_events.csv')
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data');a.add_argument('out');args=a.parse_args()
    base=Path(args.data);run(*[pd.read_parquet(base/(n+'.parquet')) for n in ('daily','five_minute','pairs')],args.out)
