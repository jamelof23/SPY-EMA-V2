"""Create SPY42 v2 five-session DIRECTION labels from strictly pre-2026 data.

No model training and no 2026 label evaluation. Uses existing chart image files.
Run: python build_v2_5day_labels.py --source SPY_daily_with_2019_warmup.csv --output output_folder
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path

HORIZON = 5
VALIDATION_YEAR = 2025
TRAIN_YEARS = set(range(2020, 2025))


def read_pre2026_history(path: Path):
    """Drop 2026+ rows BEFORE converting any price or computing any indicator."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        raw = [r for r in csv.DictReader(f) if '2019-01-01' <= r['Date'] < '2026-01-01']
    raw.sort(key=lambda r: r['Date'])
    assert len({r['Date'] for r in raw}) == len(raw), 'Duplicate dates'
    rows, ema8, ema21, atr14 = [], None, None, None
    tr_history = []
    for source in raw:
        d = date.fromisoformat(source['Date']).isoformat()
        r = {'Date': d}
        for col in ('Open', 'High', 'Low', 'Close'):
            r[col] = float(source[col])
        r['Volume'] = int(source['Volume'])
        o, h, l, c = (r[x] for x in ('Open', 'High', 'Low', 'Close'))
        assert all(math.isfinite(x) and x > 0 for x in (o, h, l, c))
        assert l <= min(o, c) <= max(o, c) <= h and r['Volume'] > 0
        prev_close = rows[-1]['Close'] if rows else c
        tr = max(h-l, abs(h-prev_close), abs(l-prev_close))
        tr_history.append(tr)
        if len(tr_history) == 14:
            atr14 = sum(tr_history) / 14
        elif len(tr_history) > 14:
            atr14 = (13*atr14 + tr)/14
        ema8 = c if ema8 is None else (2*c + 7*ema8)/9
        ema21 = c if ema21 is None else (2*c + 20*ema21)/22
        r.update(EMA8=ema8, EMA21=ema21, ATR14=atr14)
        rows.append(r)
    if not rows or rows[0]['Date'] != '2019-01-02' or rows[-1]['Date'] != '2025-12-31':
        raise ValueError('Expected audited historical source from 2019-01-02 to 2025-12-31')
    return rows


def within_atr_of_ema_today(r, level, margin):
    """Distance from the *observed* current candle's high-low range to an EMA."""
    distance = max(r['Low'] - level, level - r['High'], 0.0)
    return distance <= margin*r['ATR14']


def bear_pullback_asof(rows, t):
    """Causal context flag, NOT a future prediction or trading rule."""
    r = rows[t]
    bearish = (r['EMA8'] < r['EMA21'] and
               r['EMA21'] < rows[t-5]['EMA21'] and
               r['Close'] < r['EMA21'])
    last_3_green = sum(rows[k]['Close'] > rows[k]['Open'] for k in range(t-2, t+1))
    rebound = last_3_green >= 2 and r['Close'] > rows[t-3]['Close']
    near_either_ema = any(within_atr_of_ema_today(r, r[ema], 0.5)
                          for ema in ('EMA8', 'EMA21'))
    return int(bearish), int(rebound), int(near_either_ema), int(bearish and rebound and near_either_ema)


def produce(source: Path, output: Path):
    rows = read_pre2026_history(source)
    output.mkdir(parents=True, exist_ok=True)
    training, validation, audit = [], [], []
    setup_train_indices = []
    for t in range(42-1, len(rows)-HORIZON):
        a = rows[t]
        end = rows[t+HORIZON]
        year = int(a['Date'][:4])
        end_year = int(end['Date'][:4])
        if year in TRAIN_YEARS and end_year in TRAIN_YEARS:
            split = 'train'
        elif year == VALIDATION_YEAR and end_year == VALIDATION_YEAR:
            split = 'validation'
        else:
            continue  # Excludes 2019 and purges labels crossing into another split.
        delta = end['Close'] - a['Close']
        if delta == 0:
            continue  # Tie rule set before training: no binary outcome for an exact tie.
        direction = int(delta > 0)
        image_path = f'images/{split}/SPY_42_asof_{a["Date"]}.png'
        record = dict(sample_id=f'SPY42_{a["Date"].replace("-", "")}',
                      asof_date=a['Date'], image_path=image_path, label_up_5d=direction)
        if split == 'train':
            training.append(record)
        else:
            validation.append(record)
        bear, bounce, near, setup = bear_pullback_asof(rows, t)
        if setup and split == 'train':
            setup_train_indices.append(t)
        audit.append(dict(**record, split=split, forecast_end_date=end['Date'],
                          asof_close=f'{a["Close"]:.6f}', future_5th_close=f'{end["Close"]:.6f}',
                          outcome_5d_close_return=f'{end["Close"]/a["Close"]-1:.9f}',
                          ema8_asof=f'{a["EMA8"]:.6f}', ema21_asof=f'{a["EMA21"]:.6f}',
                          atr14_asof=f'{a["ATR14"]:.6f}',
                          bear_regime_asof=bear, green_rebound_asof=bounce,
                          near_ema_asof=near, bear_pullback_asof=setup))
    assert len(training) == 1253 and len(validation) == 245, 'Unexpected split size'
    episodes = []
    for idx in setup_train_indices:
        if not episodes or idx-episodes[-1][-1]>5:
            episodes.append([idx])
        else:
            episodes[-1].append(idx)
    def write_csv(name, records):
        with (output/name).open('w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=list(records[0]))
            wr.writeheader()
            wr.writerows(records)
    write_csv('train_labels.csv',training)
    write_csv('validation_labels.csv',validation)
    write_csv('historical_audit_through_2025.csv',audit)
    tally=lambda z:{'UP':sum(x['label_up_5d']==1 for x in z), 'DOWN':sum(x['label_up_5d']==0 for x in z)}
    summary = {'forecast_horizon_trading_sessions':5,
               'label':'UP iff close[t+5] > close[t], DOWN iff close[t+5] < close[t]; ties excluded',
               'input':'42 past daily candles with EMA8, EMA21 and volume; image pixels only',
               'train':{'count':len(training),'direction_counts':tally(training)},
               'validation':{'count':len(validation),'direction_counts':tally(validation)},
               'bear_pullback_asof_training_days':len(setup_train_indices),
               'bear_pullback_training_episodes_5day_separation':len(episodes),
               'bear_pullback_asof_validation_days':sum(x['bear_pullback_asof'] for x in audit if x['split']=='validation'),
               'test_2026':'NOT loaded, labelled, evaluated or written',
               'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(produce(args.source,args.output),indent=2))
