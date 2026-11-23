"""Add genuine traded minutes to missing regular-session five-minute buckets.

Run on the primary-only audit. Never forward-fill, overwrite primary prices or
use a secondary close sequence unless its timestamps agree on overlaps.
"""
from pathlib import Path
import hashlib, io, json, os, zipfile
import pandas as pd
import numpy as np

SECONDARY_ID="syncster-data/historical-market-data"
SECONDARY_REV="eff0dac55d843b3b83caec819efa213021a3e3e5"
CHECK_URL="https://historicaldata.net/file/stocks_sample_2022H2.zip"
CHECK_SHA="3640e34b8de32d86a3e1fd83e68e99013e5c130f72ed3a93287a2b651b24f843"
ZONE="America/New_York"
OHLCV=["open","high","low","close","volume"]

def load_secondary(days):
    import pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    import urllib.request
    if not len(days):return {}
    path=f"datasets/{SECONDARY_ID}@{SECONDARY_REV}/data/stocks/1m/S/SPY.parquet"
    pieces=[]
    with HfFileSystem().open(path,"rb",block_size=131072) as stream:
        pf=pq.ParquetFile(stream);q=pf.schema.names.index("ts")
        for i in range(pf.num_row_groups):
            stats=pf.metadata.row_group(i).column(q).statistics
            if pd.Timestamp(stats.min).year>=2026:continue
            clocks=pf.iter_batches(batch_size=4096,row_groups=[i],columns=["ts"])
            prices=pf.iter_batches(batch_size=4096,row_groups=[i],columns=["ts",*OHLCV])
            for clock in clocks:
                t=clock.column(0).to_pandas()
                if t.max()>=pd.Timestamp("2026-01-01",tz="UTC") or t.min()>days.max().tz_localize(ZONE)+pd.Timedelta(days=1):break
                page=next(prices)
                keep=t.dt.tz_convert(ZONE).dt.tz_localize(None).dt.normalize().isin(days)
                if keep.any(): pieces.append(page.filter(pa.array(keep)).to_pandas())
    alt=pd.concat(pieces).set_index("ts").sort_index().tz_convert(ZONE).tz_localize(None)
    with urllib.request.urlopen(CHECK_URL,timeout=60) as response: raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=CHECK_SHA:raise ValueError("Independent sample changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        check=pd.read_csv(z.open("minute_by_symbol/SPY_minute.csv"),usecols=["timestamp",*OHLCV],parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return {"HistoricalData.net":check,"Syncster":alt}

def repair(primary, corrected, audit, sources):
    import exchange_calendars as xc
    original, dest=Path(primary),Path(corrected)
    dest.mkdir(exist_ok=True,parents=True)
    audit=audit[audit.observed<audit.expected]
    schedule=xc.get_calendar("XNYS",start="2020-01-01",end="2025-12-31").schedule
    receipts=[]; comparisons=[]
    for f in sorted(original.glob("????-??.parquet")):
        p=pd.read_parquet(f)
        p.index=p.timestamp.dt.tz_convert(ZONE).dt.tz_localize(None)
        patched=[]
        for day in audit.index[audit.index.strftime("%Y-%m")==f.stem]:
            s=schedule.loc[day]
            left=s.open.tz_convert(ZONE).tz_localize(None);right=s.close.tz_convert(ZONE).tz_localize(None)
            reg=p.loc[(p.index>=left)&(p.index<right)]
            missing=set(pd.date_range(left,right,freq="5min",inclusive="left"))-set(reg.index.floor("5min"))
            for name,full in sources.items():
                if not missing:break
                q=full.loc[(full.index>=left)&(full.index<right),OHLCV].copy()
                q=q[(q.volume>0)&(q.low>0)&(q.high>=q[["open","close"]].max(axis=1)-.000001)&(q.low<=q[["open","close"]].min(axis=1)+.000001)]
                q=q.loc[~q.index.duplicated(keep=False)]
                shared=q.index.intersection(reg.index)
                if len(shared)==0:continue
                primary_close=reg.close
                err={}
                for lag in (-1,0,1):
                    shifted=q.close.copy()
                    shifted.index=shifted.index+pd.Timedelta(minutes=lag)
                    ix=shared.intersection(shifted.index)
                    err[lag]=((primary_close.loc[ix]-shifted.loc[ix]).abs()/primary_close.loc[ix]).mean()
                # Check the time label (a one-minute shift shouldn't improve materially)
                # and price units. With only a few overlaps also require tight median error.
                median=(q.loc[shared,"close"]/primary_close.loc[shared]-1).abs().median()
                good=(len(shared)>=4 and median<.001 and err[0] <= min(err[-1],err[1])+.000025)
                comparisons.append(dict(date=str(day.date()),source=name,overlap=len(shared),median_relative_difference=median,
                                        lag_minus1=err[-1],lag_zero=err[0],lag_plus1=err[1],accepted=good))
                if not good:continue
                found=q[q.index.floor("5min").isin(missing)].copy()
                if found.empty:continue
                found["timestamp"]=found.index.tz_localize(ZONE).tz_convert("UTC")
                patched.append(found)
                report=found.copy();report["source"]=name;receipts.append(report.reset_index(drop=True))
                missing-=set(found.index.floor("5min"))
        target=dest/f.name
        if patched:
            data=pd.concat([p,*patched]).sort_index()
            assert data.index.is_unique
            data.reset_index(drop=True).to_parquet(target,index=False,compression="zstd")
        elif not target.exists():os.link(f,target)
    ledger=(pd.concat(receipts,ignore_index=True) if receipts else pd.DataFrame(columns=[*OHLCV,"timestamp","source"]))
    return ledger,pd.DataFrame(comparisons)
