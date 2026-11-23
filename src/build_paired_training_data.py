"""Rebuild paired research inputs from the pinned, pre-2026 source partitions."""
from pathlib import Path
import argparse
import pandas as pd
from collect_spy_minutes import collect
from repair_paired_gaps import load_secondary, repair
from paired_decision_charts import build, ZONE


def prepare(root, daily):
    root=Path(root); primary=root/"minutes"; corrected=root/"corrected"
    collect(primary,workers=3)
    *_, original=build(primary,daily,root/"original")
    missing=original.index[original.expected!=original.observed]
    ledger,checks=repair(primary,corrected,original,load_secondary(missing))
    out=root/"dataset"
    _,bars,pairs,_,audit=build(corrected,daily,out)
    restored=pd.to_datetime(ledger.timestamp,utc=True).dt.tz_convert(ZONE).dt.tz_localize(None).dt.floor("5min")
    bars["secondary_source_bar"]=bars.index.isin(restored)
    pairs["secondary_in_42"]=bars.secondary_source_bar.rolling(42,min_periods=1).max().iloc[pairs.intraday_end].to_numpy().astype(bool)
    bars.to_parquet(out/"five_minute.parquet",compression="zstd")
    pairs.to_parquet(out/"pairs.parquet",compression="zstd")
    checks.to_csv(out/"overlap_check.csv",index=False)
    ledger.to_parquet(out/"recovered_minutes.parquet",compression="zstd")
    actual=pairs.groupby("split").size().to_dict()
    if actual!={"train":97792,"validation":19392} or len(audit)!=1508:
        raise ValueError(f"Rebuilt coverage differs from the prepared research data: {actual}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root");p.add_argument("daily")
    args=p.parse_args();prepare(args.root,args.daily)
