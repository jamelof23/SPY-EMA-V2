"""Development-only test of regularization and a ten-session-old daily-chart control."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from train_paired_charts import DirectionCNN, scores, atomic_json


def run(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # The base loader accepts the pre-2026 index. Restrict its input and outcomes
    # before rendering so the experiment never scores or renders a 2025 decision.
    source=Path(data);devdata=out.parent/"pre2025_only";devdata.mkdir(exist_ok=True)
    for name in ("daily","five_minute","pairs","outcomes"):
        frame=pd.read_parquet(source/(name+".parquet"))
        frame.loc[frame.index.year<=2024].to_parquet(devdata/(name+".parquet"))
    from paired_decision_charts import load_views,FEATURES
    from train_paired_charts import as_image
    daily,bars,pairs=load_views(devdata)
    labels=pd.read_parquet(devdata/"outcomes.parquet").return_30m
    pairs=pairs.loc[labels.notna() & labels.ne(0)]
    dv=daily[FEATURES].to_numpy();bv=bars[FEATURES].to_numpy()
    d_ids=pairs.daily_end.to_numpy();day_images=np.zeros((len(daily),3,48,84),dtype=np.uint8)
    for i in np.unique(np.concatenate((d_ids,d_ids-10))):
        day_images[i]=as_image(dv[i-41:i+1])
    x=np.empty((len(pairs),3,48,84),dtype=np.uint8)
    for n,row in enumerate(pairs.itertuples()):
        i=row.intraday_end
        x[n]=as_image(bv[i-41:i+1],((row.pm_high,'h','#8b76bd'),(row.pm_low,'l','#8b76bd'),(dv[row.daily_end,3],'s','#95a0ab')))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    y=(labels.loc[pairs.index].to_numpy()>0).astype('float32')
    x,di,day_images,ydevice=[torch.as_tensor(t,device=device) for t in (x,d_ids,day_images,y)]
    clocks=pairs.index;years=clocks.year
    sessions=clocks.normalize()
    summary=[]
    for year in (2022,2023,2024):
        fit_ids=np.flatnonzero(years<year);test=np.flatnonzero(years==year)
        groups=list(pd.Series(fit_ids,index=sessions[fit_ids]).groupby(level=0))
        outpred=pd.DataFrame({'up':y[test]},index=clocks[test])
        for mode in ('intraday','paired','stale_daily'):
            folder=out/f'{year}_{mode}';folder.mkdir(exist_ok=True)
            torch.manual_seed(2340+year)
            net=DirectionCNN('intraday' if mode=='intraday' else 'paired').to(device)
            for layer in net.modules():
                if isinstance(layer,nn.Dropout):layer.p=.45
            opt=torch.optim.AdamW(net.parameters(),lr=.00035,weight_decay=.025)
            history=[];existing=sorted(folder.glob('epoch_*.pt'))
            if existing:
                checkpoint=torch.load(existing[-1],map_location=device,weights_only=False)
                net.load_state_dict(checkpoint['weights']);opt.load_state_dict(checkpoint['optimizer']);history=checkpoint['history']
            def logit(ids):
                ids=torch.as_tensor(ids,device=device)
                j=di[ids]-(10 if mode=='stale_daily' else 0)
                return net(x[ids].float()/255,day_images[j].float()/255,())
            for ep in range(len(history)+1,9):
                rng=np.random.default_rng(195+ep)
                batchids=np.concatenate([rng.choice(vals.to_numpy(),min(6,len(vals)),replace=False) for _,vals in groups])
                rng.shuffle(batchids);torch.manual_seed(195+ep);net.train();training_loss=0.
                for start in range(0,len(batchids),128):
                    ids=batchids[start:start+128];opt.zero_grad(set_to_none=True);z=logit(ids)
                    loss=nn.functional.binary_cross_entropy_with_logits(z,ydevice[ids]);loss.backward();opt.step();training_loss+=loss.item()*len(ids)
                net.eval()
                with torch.no_grad():
                    prob=np.concatenate([logit(test[i:i+384]).sigmoid().cpu().numpy() for i in range(0,len(test),384)])
                row={'epoch':ep,'fit_n':len(batchids),'fit_loss':training_loss/len(batchids),'holdout':scores(y[test],prob)}
                history.append(row);tmp=folder/'saving.partial'
                torch.save(dict(weights=net.state_dict(),optimizer=opt.state_dict(),history=history,year=year,mode=mode),tmp)
                tmp.replace(folder/f'epoch_{ep:02}.pt');atomic_json(folder/'epochs.json',history)
                print('EPOCH',year,mode,json.dumps(row),flush=True)
            net.eval()
            with torch.no_grad():
                outpred[mode]=np.concatenate([logit(test[i:i+384]).sigmoid().cpu().numpy() for i in range(0,len(test),384)])
            summary.append({'holdout':year,'mode':mode,**scores(y[test],outpred[mode].to_numpy())})
            outpred.to_csv(out/f'predictions_{year}.csv');atomic_json(out/'summary.json',summary)
    print('COMPLETED',json.dumps(summary),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('data');a.add_argument('out');s=a.parse_args();run(s.data,s.out)
