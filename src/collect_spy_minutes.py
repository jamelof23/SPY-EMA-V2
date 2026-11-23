"""Extract just SPY from pinned monthly public OHLCV files. No partition from 2026 is opened."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem

FEED = "Ashutosh53/OHLCV-1m"
REV = "c3caa828b462249015462ef9a13f20271fe1b873"
COLUMNS = ["timestamp","open","high","low","close","volume"]
MONTHS = [str(p) for p in pd.period_range("2019-12","2025-12",freq="M")]

def extract_month(month, directory):
    if month not in MONTHS:
        raise ValueError("This experiment uses the pre-2026 partitions only")
    base=Path(directory)
    base.mkdir(exist_ok=True,parents=True)
    dest=base/(month+".parquet")
    if dest.is_file(): return dest
    data=[]
    uri=f"datasets/{FEED}@{REV}/data/ohlcv_{month}.parquet"
    with HfFileSystem().open(uri,"rb",block_size=262144,cache_type="readahead") as remote:
        file=pq.ParquetFile(remote)
        col=file.schema.names.index("ticker")
        for i in range(file.num_row_groups):
            stats=file.metadata.row_group(i).column(col).statistics
            if stats and not stats.min <= "SPY" <= stats.max: continue
            mask=pc.equal(file.read_row_group(i,columns=["ticker"]).column(0),"SPY")
            if pc.any(mask).as_py():
                data.append(file.read_row_group(i,columns=COLUMNS).filter(mask))
    if not data: raise ValueError(f"No SPY in {month}")
    data=pa.concat_tables(data).to_pandas().sort_values("timestamp")
    local=data.timestamp.dt.tz_convert("America/New_York")
    data=data[local.dt.strftime("%Y-%m")==month]
    before=len(data)
    data=data.drop_duplicates()
    identical=before-len(data)
    ambiguous=data.timestamp.duplicated(keep=False)
    conflicts=data.loc[ambiguous,"timestamp"].drop_duplicates()
    if len(conflicts):
        print(month,"conflicting minute timestamps excluded",len(conflicts),flush=True)
        data=data.loc[~ambiguous]
    if data.empty: raise ValueError(f"Empty local calendar month {month}")
    if identical or len(conflicts):
        (base/(month+"_duplicates.json")).write_text(json.dumps(dict(
            identical_rows_removed=identical,conflict_timestamps=[str(t) for t in conflicts]),indent=2))
    tmp=dest.with_suffix(".tmp")
    data.to_parquet(tmp,index=False,compression="zstd")
    tmp.replace(dest)
    return dest

def collect(directory,workers=6):
    def run(m):
        for i in range(3):
            try:return extract_month(m,directory)
            except Exception:
                if i==2:raise
                time.sleep(2**i)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,result in enumerate(pool.map(run,MONTHS),1):
            print(f"{n}/{len(MONTHS)}   {result.stem}",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory");p.add_argument("--workers",type=int,default=6)
    args=p.parse_args();collect(args.directory,args.workers)
