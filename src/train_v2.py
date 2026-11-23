"""Fully trainable SPY42 chart-image CNN: close(t+5) > close(t) direction.

Train on 2020–2024; develop/early-stop on 2025; never read 2026 labels.
All model input features are *pixels*, not numerical prices, dates or EMA values.
The as-of-only EMA context is evaluated as a subgroup after predictions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, log_loss, brier_score_loss

TRAIN_COUNT, VAL_COUNT = 1253,245
SHAPE = (3,224,512)
MODEL_VERSION = 'spy_v2_smallcnn_scratch_5session_v1'


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility where available, without forcing pathologically slow kernels.
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def jsonwrite(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def label_rows(repo:Path,split:str):
    with (repo/'data'/'labels'/f'{split}_labels.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    count=TRAIN_COUNT if split=='train' else VAL_COUNT
    if len(rows)!=count: raise RuntimeError('Unexpected number of '+split+' labels')
    if len({r['asof_date'] for r in rows})!=len(rows): raise RuntimeError('Duplicate as-of dates')
    for row in rows:
        year=int(row['asof_date'][:4])
        if not (2020<=year<=2024 if split=='train' else year==2025):
            raise RuntimeError('Wrong year in '+split)
        if row['label_up_5d'] not in ('0','1'): raise RuntimeError('Not a binary direction label')
        if row['image_path'] != f'images/{split}/SPY_42_asof_{row["asof_date"]}.png':
            raise RuntimeError('Unexpected path')
    return rows


class ChartDataset(Dataset):
    def __init__(self,repo:Path,data:Path,split:str):
        self.rows=label_rows(repo,split)
        self.files=[]
        self.labels=[]
        self.dates=[]
        for r in self.rows:
            p=data/'images'/split/Path(r['image_path']).name
            if not p.is_file(): raise FileNotFoundError('Missing prepared image '+str(p))
            self.files.append(p)
            self.labels.append(int(r['label_up_5d']))
            self.dates.append(r['asof_date'])
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        with Image.open(self.files[index]) as im:
            if im.size!=(SHAPE[2],SHAPE[1]):
                raise ValueError('Wrong model-image shape')
            x=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
        # No ImageNet normalization or pretrained photo features. Pixels only.
        t=torch.from_numpy(x).permute(2,0,1).float().div_(255.0)
        return t, torch.tensor(self.labels[index],dtype=torch.float32)


class SmallChartCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin,cout,k=3):
            return [nn.Conv2d(cin,cout,kernel_size=k,stride=2,padding=k//2,bias=False),
                    nn.BatchNorm2d(cout),nn.SiLU(inplace=True)]
        self.features=nn.Sequential(
            *block(3,16,k=5), *block(16,32),
            *block(32,64), *block(64,96), *block(96,128),
            nn.AdaptiveAvgPool2d((3,8)))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(0.30),
                                nn.Linear(128*3*8,128),nn.SiLU(),
                                nn.Dropout(0.25),nn.Linear(128,1))
    def forward(self,x):
        return self.head(self.features(x)).flatten()


def metrics(y,prob,threshold=0.5):
    y=np.asarray(y,dtype=np.int64); prob=np.asarray(prob,dtype=float)
    if not len(y): raise ValueError('Empty metrics')
    pred=(prob>=threshold).astype(np.int64)
    cm=confusion_matrix(y,pred,labels=[0,1]).tolist()
    prec,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'log_loss':float(log_loss(y,np.clip(prob,1e-7,1-1e-7),labels=[0,1])),
            'brier_score':float(brier_score_loss(y,prob)),
            'confusion_matrix_rows_actual_DOWN_UP_columns_predicted_DOWN_UP':cm,
            'DOWN':{'precision':float(prec[0]),'recall':float(recall[0]),'f1':float(f1[0]),
                    'count':int((y==0).sum())},
            'UP':{'precision':float(prec[1]),'recall':float(recall[1]),'f1':float(f1[1]),
                  'count':int((y==1).sum())},
            'signals':{'UP':int(pred.sum()),'DOWN':int((pred==0).sum())}}


def evaluate(model,loader,device):
    model.eval()
    y,p=[],[]
    with torch.inference_mode():
        for x,tgt in loader:
            pred=model(x.to(device,non_blocking=True))
            p.extend(pred.float().sigmoid().cpu().numpy().tolist())
            y.extend(tgt.long().cpu().numpy().tolist())
    return y,p


def assess_subgroups(y,prob,dates,repo):
    with (repo/'data'/'labels'/'historical_audit_through_2025.csv').open(newline='',encoding='utf-8') as f:
        context={r['asof_date']:r for r in csv.DictReader(f) if r['split']=='validation'}
    assert len(y)==len(prob)==len(dates)
    if any(d not in context for d in dates): raise RuntimeError('Missing as-of context')
    out={}
    for col in ('bear_regime_asof','bear_pullback_asof','near_ema_asof'):
        chosen=[i for i,d in enumerate(dates) if context[d][col]=='1']
        out[col]=metrics([y[i] for i in chosen],[prob[i] for i in chosen]) if len(chosen)>=2 else {'n':len(chosen),'notice':'Too few examples'}
    # Comparison-only EMA heuristic: no context information enters CNN.
    ema_pred=[]
    for d in dates:
        r=context[d]; bullish=float(r['ema8_asof'])>float(r['ema21_asof'])
        ema_pred.append(1.0 if bullish else 0.0)
    out['ema8_above_ema21_direction_baseline']=metrics(y,ema_pred)
    return out


def block_ci(y,prob,seed:int,block=5,n=1000):
    """Conservative approximate block-bootstrap accuracy CI for overlapping forecast horizons."""
    y=np.asarray(y,dtype=int); p=np.asarray(prob,dtype=float); size=len(y)
    groups=[np.arange(i,min(i+block,size)) for i in range(0,size,block)]
    rng=np.random.default_rng(seed+5000)
    scores=[]
    for _ in range(n):
        sampled=np.concatenate([groups[j] for j in rng.integers(0,len(groups),size=len(groups))])
        scores.append(np.mean((p[sampled]>=0.5)==y[sampled]))
    return [float(x) for x in np.percentile(scores,[2.5,97.5])]


def run(args):
    repo=args.repo_root.resolve(); data=args.data_root.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    set_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. In Colab select Runtime > Change runtime type > T4 GPU.')
    device=torch.device('cuda' if args.device=='cuda' else 'cpu')
    use_amp=(device.type=='cuda')
    train=ChartDataset(repo,data,'train'); val=ChartDataset(repo,data,'validation')
    if len(train)!=TRAIN_COUNT or len(val)!=VAL_COUNT: raise RuntimeError('Unexpected dataset counts')
    train_labels=np.asarray(train.labels)
    npos=int(train_labels.sum()); nneg=len(train_labels)-npos
    print(f'OUR GOAL: 5-trading-day UP/DOWN direction. TRAIN={len(train)} (UP {npos}, DOWN {nneg}),'
          f' VALIDATION={len(val)} (UP {sum(val.labels)}, DOWN {len(val)-sum(val.labels)})',flush=True)
    print('IMAGE ONLY: 512x224 RGB chart views; 2026 outcome labels NOT READ.',flush=True)
    model=SmallChartCNN().to(device)
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable==total
    print(f'DEVICE: {device}; ARCHITECTURE: {MODEL_VERSION}; TRAINABLE PARAMETERS={trainable:,}/{total:,} (100%)',flush=True)
    print('PRETRAINED WEIGHTS: NO; BODY-SIZE LOSS: NO.',flush=True)
    loader_kwargs={'batch_size':args.batch_size,'num_workers':args.workers,
                   'pin_memory':device.type=='cuda'}
    train_loader=DataLoader(train,shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed),**loader_kwargs)
    val_loader=DataLoader(val,shuffle=False,**loader_kwargs)
    if args.benchmark_only:
        t0=time.monotonic()
        model.train()
        steps=0
        for x,y in train_loader:
            logits=model(x.to(device)); loss=nn.functional.binary_cross_entropy_with_logits(logits,y.to(device))
            loss.backward();model.zero_grad(set_to_none=True)
            steps+=1
            if steps==args.benchmark_batches: break
        elapsed=time.monotonic()-t0
        print(f'BENCHMARK ONLY: {steps} batches, {elapsed:.2f} sec, approx {elapsed/steps*len(train_loader):.1f} sec per epoch (rough estimate; includes startup).',flush=True)
        return
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.015)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*0.05)
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    config={"model_version":MODEL_VERSION,"horizon_trading_days":5,"backbone_pretrained":False,
        "trainable_params":trainable,"total_params":total,"input_image_shape":list(SHAPE),
        "training_years":[2020,2021,2022,2023,2024],"validation_years":[2025],
        "test_2026_evaluated":False,"label_definition":"close[t+5]>close[t]",
        "train_n":len(train),"validation_n":len(val),
        "seed":args.seed,"batch_size":args.batch_size,"lr":args.lr,"max_epochs":args.epochs,
        "patience":args.patience,"image_normalization":"raw RGB / 255", 
        "checkpoint_selection":"maximum 2025 balanced accuracy; tie break with minimum log loss",
        "train_label_sha256":sha256_file(repo/'data'/'labels'/'train_labels.csv'),
        "validation_label_sha256":sha256_file(repo/'data'/'labels'/'validation_labels.csv'),
        "checkpoint_safety":"checkpoint each completed epoch; resume only with same config"}
    cfgpath=out/'training_config.json'
    if cfgpath.exists() and json.loads(cfgpath.read_text())!=config:
        raise RuntimeError('Existing output has different model/label/config; select new output directory.')
    jsonwrite(cfgpath,config)
    state_path=out/'last_checkpoint.pt'
    best_path=out/'best_direction_model.pt'
    start_epoch=1; best_score=-float('inf'); best_logloss=float('inf'); best_epoch=None; misses=0
    history=[]
    if state_path.exists():
        saved=torch.load(state_path,map_location=device,weights_only=True)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch']+1;best_score=saved['best_score'];
        best_logloss=saved['best_logloss'];best_epoch=saved['best_epoch'];misses=saved['misses'];
        history=saved['history']
        print(f'RESUMING from epoch {start_epoch} (last complete epoch {start_epoch-1}).',flush=True)
    class_weights=torch.tensor([len(train)/(2*nneg),len(train)/(2*npos)],device=device)
    for epoch in range(start_epoch,args.epochs+1):
        # Independent per-epoch ordering makes resumed and uninterrupted runs
        # use the same shuffle order for a given epoch.
        epoch_loader=DataLoader(train,shuffle=True,
            generator=torch.Generator().manual_seed(args.seed+epoch),**loader_kwargs)
        t0=time.monotonic();model.train();total_loss=0.;seen=0
        for x,y in epoch_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=use_amp):
                logits=model(x)
                raw=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                weights=torch.where(y>0.5,class_weights[1],class_weights[0])
                loss=(raw*weights).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            scaler.step(opt);scaler.update()
            total_loss+=loss.detach().item()*len(y);seen+=len(y)
        scheduler.step()
        y,p=evaluate(model,val_loader,device)
        met=metrics(y,p)
        score=met['balanced_accuracy'];logloss=met['log_loss']
        better=(score>best_score+1e-7) or (abs(score-best_score)<=1e-7 and logloss<best_logloss-1e-7)
        if better:
            best_score,best_logloss,best_epoch=score,logloss,epoch;misses=0
            # The best checkpoint contains only weights and safe numerical metadata.
            temp_best=best_path.with_suffix('.pt.partial')
            torch.save({'model':model.state_dict(),'epoch':epoch,'metric':met,
                        'model_version':MODEL_VERSION,'input_shape':SHAPE},temp_best)
            os.replace(temp_best,best_path)
        else: misses+=1
        rec={'epoch':epoch,'train_weighted_bce':total_loss/seen,
             'validation_accuracy':met['accuracy'],
             'validation_balanced_accuracy':score,'validation_logloss':logloss,
             'validation_down_recall':met['DOWN']['recall'],
             'validation_up_recall':met['UP']['recall'],
             'seconds':time.monotonic()-t0,'best_epoch_so_far':best_epoch}
        history.append(rec)
        jsonwrite(out/'history.json',history)
        temp_state=state_path.with_suffix('.pt.partial')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'best_score':best_score,'best_logloss':best_logloss,
            'best_epoch':best_epoch,'misses':misses,'history':history},temp_state)
        os.replace(temp_state,state_path)
        print(f"Epoch {epoch:02d}: loss {rec['train_weighted_bce']:.4f} | "
              f"2025 accuracy {met['accuracy']:.1%} | balanced accuracy {score:.1%} | "
              f"DOWN recall {met['DOWN']['recall']:.1%} | UP recall {met['UP']['recall']:.1%} | "
              f"logloss {logloss:.4f} | {rec['seconds']:.1f}s | best epoch {best_epoch}",flush=True)
        if misses>=args.patience:
            print(f'EARLY STOP: {misses} successive epochs without better 2025 balanced accuracy/logloss.',flush=True)
            break
    if not best_path.exists(): raise RuntimeError('No completed best checkpoint')
    saved_best=torch.load(best_path,map_location=device,weights_only=True)
    model.load_state_dict(saved_best['model'])
    y,p=evaluate(model,val_loader,device)
    met=metrics(y,p)
    met['accuracy_95pct_block_bootstrap_ci']=block_ci(y,p,args.seed)
    met['subgroups_2025_asof_only']=assess_subgroups(y,p,val.dates,repo)
    met['always_up_baseline']=metrics(y,np.ones(len(y)))
    met['always_down_baseline']=metrics(y,np.zeros(len(y)))
    met['checkpoints']={'best_epoch':best_epoch,'actual_epochs':len(history),'model_v2':MODEL_VERSION,
                        'all_cnn_layers_trained':True}
    met['test_2026_evaluated']=False
    for threshold in (0.55,0.60,0.65):
        confidence_indices=[i for i,q in enumerate(p) if q>=threshold or q<=1-threshold]
        met[f'predeclared_confidence_{threshold}']=(
            {'n':len(confidence_indices),'coverage':len(confidence_indices)/len(y),
             'accuracy':float(np.mean([(p[i]>=0.5)==bool(y[i]) for i in confidence_indices])),
             'UP_signals':sum(p[i]>=threshold for i in confidence_indices),
             'DOWN_signals':sum(p[i]<=1-threshold for i in confidence_indices)}
            if confidence_indices else {'n':0,'coverage':0.0})
    jsonwrite(out/'validation_2025_metrics.json',met)
    with (out/'validation_2025_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['asof_date','label_up_5d','p_up_5d','pred_up_5d'])
        for d,truth,prob in zip(val.dates,y,p):w.writerow([d,truth,round(prob,7),int(prob>=0.5)])
    print('\n=== COMPLETED 2025 VALIDATION REPORT (development only) ===',flush=True)
    print(json.dumps(met,indent=2),flush=True)
    print('Saved to',out,flush=True)
    print('2026 outcome labels were NOT touched.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=0.0005)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--benchmark-batches',type=int,default=10)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.patience<1:raise ValueError('Invalid hyperparameters')
    run(args)

if __name__=='__main__': main()
