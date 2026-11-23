"""Repeat the target-before-stop comparison on three nonoverlapping validation years."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from paired_baselines import features


def run(daily,bars,pairs,events,dest):
    events=events[events.index.year<=2024]
    p=pairs.loc[events.index];clock,_,ctx=features(daily,bars,p)
    scale=(bars.high-bars.low).rolling(12).mean().to_numpy()[p.intraday_end].clip(.02)
    geometry=clock.copy()
    geometry['to_target']=(events.target-events.signal)/scale
    geometry['to_stop']=(events.signal-events.stop)/scale
    full=geometry.join(ctx.drop(columns=clock.columns))
    Y=(events.result=='target').astype(int);day=Y.index.normalize()
    weights=1/Y.groupby(day).transform('size')
    output=[];preds=[]
    for year in (2022,2023,2024):
        train=Y.index.year<year;test=Y.index.year==year
        w=weights[train]/weights[train].mean()
        # Same model and settings on geometry alone and with chart features isolates feature value.
        specs=[('distance_linear',geometry,'linear'),('distance_tree',geometry,'tree'),('distance_chart_tree',full,'tree')]
        frame=pd.DataFrame({'up':Y[test],'weight':weights[test]})
        for name,X,typ in specs:
            if typ=='linear':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=300))
                model.fit(X[train],Y[train],logisticregression__sample_weight=w)
            else:
                model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=3,early_stopping=False,random_state=41)
                model.fit(X[train],Y[train],sample_weight=w)
            pr=model.predict_proba(X[test])[:,1];frame[name]=pr
            output.append(dict(year=year,model=name,decisions=int(test.sum()),sessions=Y[test].index.normalize().nunique(),
                auc=roc_auc_score(Y[test],pr,sample_weight=weights[test]),log_loss=log_loss(Y[test],pr,sample_weight=weights[test],labels=[0,1])))
        preds.append(frame)
    result=pd.DataFrame(output);pred=pd.concat(preds)
    t=pred.up.to_numpy()
    prob=lambda a:np.where(t,pred[a],1-pred[a])
    delta=np.log(prob('distance_tree'))-np.log(prob('distance_chart_tree'))
    by=pd.Series(delta,index=pred.index).groupby(pred.index.normalize()).mean()
    rng=np.random.default_rng(56);view=by.to_numpy();bs=view[rng.integers(len(view),size=(4000,len(view)))].mean(1)
    dest=Path(dest);dest.mkdir(exist_ok=True,parents=True);result.to_csv(dest/'walk_forward.csv',index=False)
    pred.to_csv(dest/'walk_forward_predictions.csv')
    ci=pd.DataFrame([{'num_days':len(view),'chart_minus_geometry_loss':view.mean(),'lo':np.quantile(bs,.025),'hi':np.quantile(bs,.975)}])
    ci.to_csv(dest/'walk_forward_uncertainty.csv',index=False)
    print(result.to_string(index=False));print(ci.to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data');a.add_argument('events_csv');a.add_argument('out');args=a.parse_args()
    base=Path(args.data);events=pd.read_csv(args.events_csv,index_col='decision_et',parse_dates=True)
    run(*[pd.read_parquet(base/(n+'.parquet')) for n in ('daily','five_minute','pairs')],events,args.out)
