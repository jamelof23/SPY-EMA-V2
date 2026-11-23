"""Prepare SPY42 v2's PRE-2026 chart images for a five-session direction model.

Reads the committed 2020-2024 training and 2025 development label lists.
Never opens a 2026 target, a 2026 input image, or the source manifest's labels.
Source charts were rendered prior to their respective forecast dates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

from PIL import Image

ORIGINAL_DATASET = "https://github.com/jamelof23/SPY/releases/download/spy42-data-2026-11-22-v1/SPY_42_dataset.zip"
ORIGINAL_SHA = "16a8f15d81ffeb71f1c96b51bbeb978b5bdba9a4c24ca469db8c94d60094e99d"
NEW_TAG = "spy42-v2-images-v1"
NEW_NAME = "SPY42_v2_trainval_images.zip"
SOURCE_SIZE_MAX = 350_000_000
VIEW_W, VIEW_H = 512, 224
PRICE_BOX = (0.047, 0.175, 0.912, 0.695)
VOLUME_BOX = (0.047, 0.742, 0.912, 0.895)
DATASET_FILES = {"train": 1253, "validation": 245}


def read_labels(repo: Path):
    labels = {}
    for split, expected in DATASET_FILES.items():
        p = repo / "data" / "labels" / f"{split}_labels.csv"
        with p.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            raise ValueError(f"{split} labels unexpectedly changed: {len(rows)}")
        if len({r['asof_date'] for r in rows}) != len(rows):
            raise ValueError("Duplicate forecast as-of date")
        for row in rows:
            year = int(row["asof_date"][:4])
            if (split == "train" and not (2020 <= year <= 2024)) or (split == "validation" and year != 2025):
                raise ValueError("2026/incorrect year present in labels")
            if row['label_up_5d'] not in ('0','1'):
                raise ValueError("Expected strictly binary five-session labels")
            expected_image = f'images/{split}/SPY_42_asof_{row["asof_date"]}.png'
            if row['image_path'] != expected_image:
                raise ValueError("Unexpected chart path in label list")
        labels[split] = rows
    return labels


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def download_public_archive(repo_slug: str, archive: Path):
    """First try the v2 train/validation-only release, fall back to pinned v1 image source."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if zipfile.is_zipfile(archive):
            print(f"Reusing existing local ZIP: {archive}", flush=True)
            return
        print('Removing incomplete download before retry:',archive,flush=True)
        archive.unlink()
    v2_base = f"https://github.com/{repo_slug}/releases/download/{NEW_TAG}"
    errors = []
    options = [(f"{v2_base}/{NEW_NAME}", None), (ORIGINAL_DATASET, ORIGINAL_SHA)]
    for url, expected_sha in options:
        if expected_sha is None:
            # The v2 GitHub Release publishes its independently verified SHA256.
            # Do not trust a large downloaded archive without checking metadata.
            try:
                metareq=urllib.request.Request(f"{v2_base}/dataset_metadata.json",
                            headers={'User-Agent':'SPY42-v2-reproducible-training/1.0'})
                with urllib.request.urlopen(metareq,timeout=45) as response:
                    if int(response.headers.get('Content-Length','0') or '0') > 10000:
                        raise RuntimeError('Unexpectedly large release manifest')
                    metadata=json.loads(response.read(10001).decode('utf-8'))
                if (metadata.get('asset_name')!=NEW_NAME or
                    metadata.get('train_images')!=1253 or
                    metadata.get('validation_images')!=245 or
                    metadata.get('2026_test_images_included') is not False or
                    metadata.get('2026_test_labels_included') is not False or
                    len(str(metadata.get('sha256','')))!=64):
                    raise RuntimeError('Unexpected v2 release metadata')
                expected_sha=metadata['sha256']
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as e:
                errors.append(f'V2 release not ready: {type(e).__name__} {e}')
                continue
        candidate = archive.with_suffix('.part')
        try:
            print("Fetching chart images from", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent":"SPY42-v2-reproducible-training/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response, candidate.open('wb') as output:
                size=0
                for block in iter(lambda:response.read(1024*1024),b''):
                    size += len(block)
                    if size > SOURCE_SIZE_MAX:
                        raise RuntimeError("Image archive exceeds the allowed bound")
                    output.write(block)
            if sha256(candidate)!=expected_sha:
                raise RuntimeError("Pinned original image archive digest mismatch")
            if not zipfile.is_zipfile(candidate):
                raise ValueError("Downloaded file is not a ZIP")
            candidate.replace(archive)
            print("Downloaded chart archive (sha256):",sha256(archive),flush=True)
            return
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError,ValueError,RuntimeError) as e:
            errors.append(f"{url}: {type(e).__name__} {e}")
            candidate.unlink(missing_ok=True)
    raise RuntimeError("Could not get training images: " + " | ".join(errors))


def render_model_view(img: Image.Image) -> Image.Image:
    """Same two-panel crop as v1; resized to 512x224 for the v2 CNN."""
    img=img.convert("RGB")
    W,H=img.size
    def crop(box):
        return img.crop(tuple(round(z*(W if i%2==0 else H)) for i,z in enumerate(box)))
    price=crop(PRICE_BOX)
    volume=crop(VOLUME_BOX)
    price_h=round(VIEW_H*price.height/(price.height+volume.height))
    out=Image.new('RGB',(VIEW_W,VIEW_H),'black')
    out.paste(price.resize((VIEW_W,price_h),Image.Resampling.LANCZOS),(0,0))
    out.paste(volume.resize((VIEW_W,VIEW_H-price_h),Image.Resampling.LANCZOS),(0,price_h))
    return out


def prepare(repo: Path, dest: Path, archive: Path):
    labels=read_labels(repo)
    dest.mkdir(parents=True, exist_ok=True)
    image_root=dest/'images'
    missing=0
    with zipfile.ZipFile(archive) as zipped:
        names=set(zipped.namelist())
        for split,rows in labels.items():
            (image_root/split).mkdir(parents=True,exist_ok=True)
            for idx,row in enumerate(rows,1):
                relative=row['image_path']
                if relative.startswith("images/test") or '2026' in relative:
                    raise ValueError("2026 image access is prohibited")
                desired=image_root/split/Path(relative).name
                # A previous runtime may have interrupted an image write.
                # Always decode cached files fully before reusing them.
                if desired.exists():
                    try:
                        with Image.open(desired) as cached:
                            cached.load()
                            if cached.size != (VIEW_W, VIEW_H):
                                raise ValueError('Wrong cached image shape')
                        continue
                    except (OSError,ValueError):
                        print('Repairing incomplete cached image:',desired.name,flush=True)
                        desired.unlink(missing_ok=True)
                member_candidates=(f'spy42_v2/{relative}',f'spy_42_dataset/{relative}')
                matching=[x for x in member_candidates if x in names]
                if len(matching)!=1:
                    missing+=1
                    if missing<=5: print('Missing:',relative,flush=True)
                    continue
                info=zipped.getinfo(matching[0])
                if info.file_size>8*1024*1024:
                    raise ValueError("Unexpectedly large chart image member")
                with zipped.open(info) as stream, Image.open(stream) as original:
                    if original.size != (1920,1080):
                        raise ValueError(f"Unexpected source chart dimensions {original.size}")
                    processed=render_model_view(original)
                    temp=desired.with_name(desired.name+'.partial')
                    try:
                        processed.save(temp,format='PNG',compress_level=5)
                        with Image.open(temp) as check:
                            check.load()
                            if check.size != (VIEW_W, VIEW_H):
                                raise ValueError('Rendered image failed size verification')
                        temp.replace(desired)
                    finally:
                        temp.unlink(missing_ok=True)
                if idx%200==0 or idx==len(rows):
                    print(f"Prepared {split} charts: {idx}/{len(rows)}",flush=True)
    if missing:
        raise RuntimeError(f"{missing} image(s) missing; review source dataset release")
    for split,rows in labels.items():
        if any(not (image_root/split/Path(r['image_path']).name).is_file() for r in rows):
            raise AssertionError('Prepared image count mismatch')
    meta=dict(horizon_trading_days=5, train_samples=len(labels['train']),
              validation_samples=len(labels['validation']),
              image_shape=[3,VIEW_H,VIEW_W],
              training_images='2020-2024',validation_images='2025',
              test_2026='NOT accessed', source_archive_sha256=sha256(archive))
    (dest/'prepare_metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
    return meta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--archive',type=Path,default=None,
                    help='Already-downloaded trusted archive for local tests or offline training.')
    ap.add_argument('--repo-slug',default='jamelof23/SPY-EMA-V2')
    args=ap.parse_args()
    archive=args.archive or args.dest/'downloaded_images.zip'
    if args.archive is None:
        download_public_archive(args.repo_slug,archive)
    prepare(args.repo_root,args.dest,archive)

if __name__=='__main__':
    main()
