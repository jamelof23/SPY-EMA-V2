"""Train a small learned convolutional image baseline on the existing paired PNGs.

No GPU or deep-learning framework is required. All convolution filters and the
classification head receive gradients. Checkpoints contain Adam state and weights.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, log_loss
from numpy.lib.stride_tricks import sliding_window_view


def load_panel(z,name):
    with Image.open(io.BytesIO(z.read(name))) as im:
        return np.asarray(im.convert("RGB").resize((42,24),Image.Resampling.BOX))


def images(index,archives,folder):
    dest=folder/"images.npy"
    keys=folder/"dates.npy"
    if dest.exists() and keys.exists() and np.array_equal(np.load(keys),index.index.asi8):
        return np.load(dest,mmap_mode="r")
    tmp=folder/"images.partial.npy"
    stack=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.uint8,shape=(len(index),2,24,42,3))
    for yr in sorted(set(index.index.year)):
        chosen=np.flatnonzero(index.index.year==yr)
        cache={}
        with zipfile.ZipFile(archives/f"SPY_chart_pairs_{yr}.zip") as z:
            for n,row in zip(chosen,index.iloc[chosen].itertuples()):
                if row.daily_end not in cache:
                    cache[row.daily_end]=load_panel(z,f"daily/{row.daily_end}.png")
                stack[n,0]=cache[row.daily_end]
                stack[n,1]=load_panel(z,f"five_minute/{row.intraday_end}.png")
        print("Loaded PNGs",yr,len(chosen),flush=True)
    stack.flush();del stack;tmp.replace(dest);np.save(keys,index.index.asi8)
    return np.load(dest,mmap_mode="r")


def model_params(count,seed):
    g=np.random.default_rng(seed);p={}
    for i in range(count):
        p[f"k{i}"]=g.normal(0,.18,(27,8)).astype('float32')
        p[f"b{i}"]=np.full(8,.01,dtype="float32")
    p["w"]=g.normal(0,.045,(count*6*8,)).astype('float32')
    p["bias"]=np.zeros(1,dtype="float32")
    return p


def forward(batch,p,panels):
    n=len(batch);chunks=[];cache=[]
    for i,col in enumerate(panels):
        x=(batch[:,col].astype('float32')-np.array([19,23,34],dtype='float32'))/128
        view=sliding_window_view(x,(3,3),axis=(1,2))[:,::2,::2]
        # Keep ten by eighteen convolution locations, partitioned into six regions.
        view=np.ascontiguousarray(view[:,:10,:18]).reshape(n*180,27)
        a=view@p[f"k{i}"]+p[f"b{i}"]
        act=np.maximum(a,0).reshape(n,2,5,3,6,8).mean(axis=(2,4))
        chunks.append(act.reshape(n,-1));cache.append((view,a))
    feat=np.concatenate(chunks,axis=1)
    return expit(feat@p["w"]+p["bias"]),feat,cache


def update(x,y,p,panels,m,v,step):
    pr,feat,cache=forward(x,p,panels)
    dz=(pr-y)/len(y);g={"w":feat.T@dz,"bias":np.array([dz.sum()])}
    df=dz[:,None]*p["w"]
    for i,(view,a) in enumerate(cache):
        cell=df[:,i*48:(i+1)*48].reshape(len(y),2,1,3,1,8)/30
        delta=np.broadcast_to(cell,(len(y),2,5,3,6,8)).reshape(-1,8)*(a>0)
        g[f"k{i}"]=view.T@delta;g[f"b{i}"]=delta.sum(axis=0)
    for key in p:
        if key.startswith('k') or key=='w':g[key]+=0.0005*p[key]
        m[key]=.9*m[key]+.1*g[key];v[key]=.999*v[key]+.001*g[key]**2
        p[key]-=.002*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
    return pr


def metrics(y,p):
    return dict(n=len(y),accuracy=accuracy_score(y,p>=.5),balanced_accuracy=balanced_accuracy_score(y,p>=.5),
                auc=roc_auc_score(y,p),log_loss=log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]))


def predict(x,ids,p,cols):
    return np.concatenate([forward(x[ids[s:s+256]],p,cols)[0] for s in range(0,len(ids),256)])


def run(index,outcomes,archives,out,epochs=4):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    y=outcomes.return_30m.reindex(index.index);keep=y.notna()&y.ne(0)
    index=index.loc[keep];y=(y[keep].to_numpy()>0).astype('float32')
    assert index.index.max().year==2025
    x=images(index,Path(archives),out)
    idx=index.index;fit=np.flatnonzero(idx.year<=2023);dev=np.flatnonzero(idx.year==2024);val=np.flatnonzero(idx.year==2025)
    report={};predictions=pd.DataFrame({'up':y[val]},index=idx[val])
    for name,cols in [('five_minute',[1]),('paired',[1,0])]:
        folder=out/name;folder.mkdir(exist_ok=True)
        p=model_params(len(cols),123);m={k:np.zeros_like(w) for k,w in p.items()};v={k:np.zeros_like(w) for k,w in p.items()}
        history=[];step=0;previous=sorted(folder.glob('epoch_*.npz'))
        if previous:
            with np.load(previous[-1]) as c:
                p={k:c[k] for k in p};m={k:c['m_'+k] for k in p};v={k:c['v_'+k] for k in p}
                step=int(c['step']);history=json.loads(str(c['history']))
        for epoch in range(len(history)+1,epochs+1):
            ids=np.random.default_rng(123+epoch).permutation(fit);prfit=[];yy=[]
            for j in range(0,len(ids),192):
                batch=ids[j:j+192];step+=1
                prfit.append(update(x[batch],y[batch],p,cols,m,v,step));yy.append(y[batch])
            mt=metrics(np.concatenate(yy),np.concatenate(prfit));md=metrics(y[dev],predict(x,dev,p,cols))
            row={'epoch':epoch,'fit':mt,'dev_2024':md};history.append(row)
            f=folder/f'epoch_{epoch:02}.npz';tmp=folder/'saving.npz'
            np.savez_compressed(tmp,**p,**{'m_'+k:a for k,a in m.items()},**{'v_'+k:a for k,a in v.items()},step=step,history=json.dumps(history))
            tmp.replace(f);(folder/'epochs.json').write_text(json.dumps(history,indent=2))
            print('EPOCH',name,json.dumps(row),flush=True)
        best=min(history,key=lambda r:r['dev_2024']['log_loss'])['epoch']
        with np.load(folder/f'epoch_{best:02}.npz') as c:bp={k:c[k] for k in p}
        prob=predict(x,val,bp,cols);report[name]={'selected_epoch':best,'validation_2025':metrics(y[val],prob)}
        predictions[name]=prob
        predictions.to_csv(out/'predictions_2025.csv');(out/'summary.json').write_text(json.dumps(report,indent=2))
        print('LOCKED_2025',name,json.dumps(report[name]),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('data');ap.add_argument('archives');ap.add_argument('out');ap.add_argument('--epochs',type=int,default=4);a=ap.parse_args()
    base=Path(a.data);run(pd.read_parquet(base/'pairs.parquet'),pd.read_parquet(base/'outcomes.parquet'),a.archives,a.out,a.epochs)
