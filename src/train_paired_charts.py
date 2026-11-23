"""Train and compare two-view chart CNNs on next-30-minute SPY direction."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

import paired_decision_charts as charts


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 12, 5, stride=2, padding=2), nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(),
        )


class DirectionCNN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.intraday = Encoder()
        self.daily = None if mode == "intraday" else Encoder()
        self.levels = mode == "paired_levels"
        size = 192 * (1 + int(self.daily is not None)) + 6 * int(self.levels)
        self.head = nn.Sequential(nn.Linear(size, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, intraday, daily, levels):
        features = [self.intraday(intraday)]
        if self.daily is not None:
            features.append(self.daily(daily))
        if self.levels:
            features.append(levels)
        return self.head(torch.cat(features, dim=1)).flatten()


def as_image(values, guides=()):
    im = charts.compact(values, guides).resize((84, 48), Image.Resampling.BOX)
    return np.asarray(im).transpose(2, 0, 1)


def prepare(data, output):
    daily, bars, pairs = charts.load_views(data)
    labels = pd.read_parquet(data / "outcomes.parquet").return_30m.reindex(pairs.index)
    assert pairs.index.max().year == 2025 and pairs.index.is_unique
    pairs = pairs.loc[labels.notna() & labels.ne(0)].copy()
    target = (labels.reindex(pairs.index).to_numpy() > 0).astype(np.float32)
    dv, bv = [frame[charts.FEATURES].to_numpy() for frame in (daily,bars)]
    di, bi = pairs.daily_end.to_numpy(), pairs.intraday_end.to_numpy()
    assert (daily.index[di] < pairs.index.normalize()).all()
    assert (bars.index[bi] + pd.Timedelta(minutes=5) == pairs.index).all()
    day_images = np.zeros((len(daily),3,48,84),dtype=np.uint8)
    for i in np.unique(di):
        day_images[i] = as_image(dv[i-41:i+1])
    # Cache inputs once for comparable models. Frames are normalized using only visible bars.
    intraday = np.empty((len(pairs),3,48,84),dtype=np.uint8)
    for n,row in enumerate(pairs.itertuples()):
        i = row.intraday_end
        guides = ((row.pm_high,"high","#8b76bd"),(row.pm_low,"low","#8b76bd"),
                  (dv[row.daily_end,3],"prior","#95a0ab"))
        intraday[n] = as_image(bv[i-41:i+1],guides)
        if n % 25000 == 0: print("Rendered",n,"of",len(pairs),flush=True)
    scale = bars.close.rolling(42).std().to_numpy()[bi].clip(.05)
    ref = bv[bi,3]
    levels = np.column_stack([(v-ref)/scale for v in (dv[di,3],dv[di,5],dv[di,6],
                              pairs.pm_high.to_numpy(),pairs.pm_low.to_numpy(),bv[bi,6])])
    levels = np.nan_to_num(levels).clip(-8,8).astype(np.float32)/8
    return intraday,day_images,di,levels,target,pairs.index


def scores(y,p):
    return dict(n=int(len(y)),accuracy=float(accuracy_score(y,p>=.5)),
                balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)),
                auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))


def atomic_json(path,obj):
    tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)


def train(args):
    data,out = Path(args.data),Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    print("Device",device,torch.cuda.get_device_name() if device.type=="cuda" else "",flush=True)
    x,d,j,l,y,dates=prepare(data,out)
    # Store compact bytes on-device and decode only each minibatch.
    x,d,j,l,y = [torch.as_tensor(a,device=device) for a in (x,d,j,l,y)]
    fit=np.flatnonzero(dates.year<=2023)
    dev=np.flatnonzero(dates.year==2024)
    val=np.flatnonzero(dates.year==2025)
    assert len(fit) and len(dev) and len(val)
    old_y=y.detach().cpu().numpy()
    base=float(old_y[fit].mean())
    results={"horizon_minutes":30,"device":str(device),"fit":len(fit),"dev":len(dev),"validation":len(val),
             "base_up_frequency":base,"baseline_2025":scores(old_y[val],np.full(len(val),base)),"models":{}}
    pred=pd.DataFrame(index=dates[val]);pred["up"]=old_y[val]
    for mode in args.modes:
        torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
        model=DirectionCNN(mode).to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.002)
        loss_fn=nn.BCEWithLogitsLoss()
        folder=out/mode;folder.mkdir(exist_ok=True)
        history=[];best=np.inf;best_epoch=0
        checkpoints=sorted(folder.glob("epoch_*.pt"))
        if checkpoints:
            state=torch.load(checkpoints[-1],map_location=device,weights_only=False)
            assert state["mode"]==mode and state["seed"]==args.seed
            model.load_state_dict(state["weights"]);optim.load_state_dict(state["optimizer"])
            history=state["history"];best=min(r["dev"]["log_loss"] for r in history)
            best_epoch=min(history,key=lambda r:r["dev"]["log_loss"])["epoch"]
        def logits(ids):
            ids=torch.as_tensor(ids,dtype=torch.long,device=device)
            return model(x[ids].float()/255,d[j[ids]].float()/255,l[ids])
        def predict(ids):
            model.eval()
            with torch.no_grad():
                return np.concatenate([logits(b).sigmoid().cpu().numpy() for b in np.array_split(ids,max(1,int(np.ceil(len(ids)/512))))])
        for ep in range(len(history)+1,args.epochs+1):
            order=np.random.default_rng(args.seed+ep).permutation(fit)
            torch.manual_seed(args.seed+ep)
            model.train();total=0.;correct=0
            for b in np.array_split(order,int(np.ceil(len(order)/args.batch))):
                t=y[torch.as_tensor(b,device=device)];optim.zero_grad(set_to_none=True)
                z=logits(b);loss=loss_fn(z,t);loss.backward();optim.step()
                total+=loss.item()*len(b);correct+=int(((z.detach()>=0)==(t>=.5)).sum().item())
            review=scores(old_y[dev],predict(dev))
            row={"epoch":ep,"fit_loss":total/len(fit),"fit_accuracy":correct/len(fit),"dev":review}
            history.append(row)
            file=folder/f"epoch_{ep:02}.pt";tmp=file.with_suffix(".partial")
            torch.save({"mode":mode,"seed":args.seed,"weights":model.state_dict(),"optimizer":optim.state_dict(),"history":history},tmp);tmp.replace(file)
            if review["log_loss"]<best:best=review["log_loss"];best_epoch=ep
            atomic_json(folder/"epochs.json",history)
            print("EPOCH",mode,json.dumps(row),flush=True)
        state=torch.load(folder/f"epoch_{best_epoch:02}.pt",map_location=device,weights_only=False)
        model.load_state_dict(state["weights"])
        p=predict(val);pred[mode]=p
        results["models"][mode]={"selected_epoch":best_epoch,"dev_log_loss":best,"2025":scores(old_y[val],p)}
        pred.to_csv(out/"predictions_2025.csv");atomic_json(out/"summary.json",results)
        print("LOCKED_2025",mode,json.dumps(results["models"][mode]),flush=True)
    print("COMPLETE",json.dumps(results),flush=True)


if __name__ == "__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--data",required=True);a.add_argument("--out",required=True)
    a.add_argument("--modes",nargs="+",choices=["intraday","paired","paired_levels"],default=["intraday","paired","paired_levels"])
    a.add_argument("--epochs",type=int,default=4);a.add_argument("--batch",type=int,default=256);a.add_argument("--seed",type=int,default=42)
    train(a.parse_args())
