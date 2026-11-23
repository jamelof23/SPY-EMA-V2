"""Create a dataset release containing ONLY pre-2026 train/validation chart PNGs.

Run inside the new SPY-EMA-V2 repository, using the old pinned SPY dataset
archive as a source. Never copy or inspect the old archive's manifest or 2026
images. The resulting ZIP contains 1,498 PNG files and no private/test labels.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, zipfile
from pathlib import Path

TRAIN_N=1253
VAL_N=245


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    repo=args.repo.resolve()
    wanted=[]
    for split,expected in (('train',TRAIN_N),('validation',VAL_N)):
        with (repo/'data'/'labels'/f'{split}_labels.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        if len(rows)!=expected:raise ValueError('Unexpected label count')
        for r in rows:
            year=int(r['asof_date'][:4])
            if (split=='train' and not 2020<=year<=2024) or (split=='validation' and year!=2025):
                raise ValueError('Test year in dataset selection')
            rel=f'images/{split}/SPY_42_asof_{r["asof_date"]}.png'
            if r['image_path']!=rel:raise ValueError('Unexpected chart input path')
            wanted.append(rel)
    if len(wanted)!=len(set(wanted)) or len(wanted)!=TRAIN_N+VAL_N:
        raise ValueError('Unexpected or duplicated image selection')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.source) as source, zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as target:
        names=set(source.namelist())
        for i,rel in enumerate(sorted(wanted),1):
            member='spy_42_dataset/'+rel
            if member not in names:raise ValueError('Image missing '+member)
            info=source.getinfo(member)
            if info.file_size>8*1024*1024:raise ValueError('Unexpected image size')
            content=source.read(member)
            if not content.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Image is not a valid PNG header')
            output_member='spy42_v2/'+rel
            zinfo=zipfile.ZipInfo(output_member,date_time=(2020,1,1,0,0,0))
            zinfo.compress_type=zipfile.ZIP_STORED
            zinfo.external_attr=0o644 << 16
            target.writestr(zinfo,content)
            if i%250==0: print('Packed',i,'/',len(wanted),flush=True)
    meta={'asset_name':args.output.name,'sha256':sha256(args.output),
          'size':args.output.stat().st_size,
          'train_images':TRAIN_N,'validation_images':VAL_N,
          'forecast_horizon_sessions':5,
          '2026_test_images_included':False,'2026_test_labels_included':False,
          'source':'Pinned 2026-11-22 SPY dataset release',
          'label_definition':'1 iff close[t+5]>close[t]'}
    (args.output.parent/'dataset_metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
