"""Compare causal non-image benchmarks for 30-minute SPY direction."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(daily, bars, pairs):
    i,d=pairs.intraday_end.to_numpy(),pairs.daily_end.to_numpy()
    idx=pairs.index
    close=bars.close.to_numpy()[i]
    daily_close=daily.close.to_numpy()[d]
    minute=idx.hour*60+idx.minute-570
    time=pd.DataFrame({"sin":np.sin(minute/390*2*np.pi),"cos":np.cos(minute/390*2*np.pi)},index=idx)
    five=time.copy()
    for lag in (1,3,6,12,24,41):
        five[f"ret_{lag}"]=close/bars.close.to_numpy()[i-lag]-1
    scale=(bars.high-bars.low).rolling(12).mean().to_numpy()[i].clip(.02)
    for k in ("ema8","ema21","open","high","low"):
        five[k]=(bars[k].to_numpy()[i]-close)/scale
    five["vol_ratio"]=bars.volume.to_numpy()[i]/bars.volume.rolling(12).mean().to_numpy()[i].clip(1)
    both=five.copy()
    for lag in (1,5,20,41):
        both[f"daily_ret_{lag}"]=daily_close/daily.close.to_numpy()[d-lag]-1
    for k in ("close","ema8","ema21","open","high","low"):
        both[f"daily_{k}"]=(daily[k].to_numpy()[d]-close)/scale
    both["pm_high"]=(pairs.pm_high-close)/scale
    both["pm_low"]=(pairs.pm_low-close)/scale
    return time,five,both


def metrics(y,p):
    return {"n":len(y),"accuracy":accuracy_score(y,p>=.5),"balanced_accuracy":balanced_accuracy_score(y,p>=.5),
            "auc":roc_auc_score(y,p),"log_loss":log_loss(y,p,labels=[0,1])}


def run(daily,bars,pairs,labels,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    target=labels.return_30m.reindex(pairs.index)
    take=target.notna() & target.ne(0)
    y=(target[take]>0).astype(int);pairs=pairs.loc[take]
    t,f,x=features(daily,bars,pairs)
    fit=y.index.year<=2023
    splits={"dev_2024":y.index.year==2024,"validation_2025":y.index.year==2025}
    methods={"time_logistic":(t,make_pipeline(StandardScaler(),LogisticRegression(max_iter=200))),
             "five_logistic":(f,make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=300))),
             "paired_logistic":(x,make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=300))),
             "paired_tree":(x,HistGradientBoostingClassifier(learning_rate=.05,max_iter=65,max_leaf_nodes=9,min_samples_leaf=150,l2_regularization=2,early_stopping=False,random_state=42))}
    result={"n_fit":int(sum(fit)),"label":"positive 30-minute return; ties omitted","splits":{}}
    pred=pd.DataFrame({"up":y,"return_30m":target[take],"prior":y[fit].mean()})
    for name,(frame,model) in methods.items():
        model.fit(frame[fit],y[fit]);pred[name]=model.predict_proba(frame)[:,1]
        joblib.dump(model,out/f"{name}.joblib")
    for split,sel in splits.items():
        frame=pred[sel]
        result["splits"][split]={k:metrics(frame.up,frame[k]) for k in ["prior",*methods]}
        # Bootstrap by session to retain dependence among overlapping 30-minute labels.
        a,b=frame.paired_logistic.to_numpy(),frame.five_logistic.to_numpy()
        true=frame.up.to_numpy()
        difference=(np.log(np.where(true==1,b,1-b))-np.log(np.where(true==1,a,1-a)))
        grouped=pd.Series(difference,index=frame.index).groupby(frame.index.normalize()).agg(['sum','count']).to_numpy()
        rng=np.random.default_rng(575);draw=grouped[rng.integers(len(grouped),size=(2000,len(grouped)))]
        boot=draw[:,:,0].sum(1)/draw[:,:,1].sum(1)
        result["splits"][split]["daily_context_log_loss_difference"]={"paired_minus_five":float(difference.mean()),"session_bootstrap_95":np.quantile(boot,[.025,.975]).tolist()}
    pred.loc[pred.index.year>=2024].to_csv(out/'predictions.csv')
    (out/'benchmarks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("data");p.add_argument("out");args=p.parse_args()
    frames=[pd.read_parquet(Path(args.data)/(n+'.parquet')) for n in ('daily','five_minute','pairs','outcomes')]
    run(*frames,args.out)
