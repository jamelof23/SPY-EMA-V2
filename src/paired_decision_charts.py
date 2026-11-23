"""Build and render SPY daily + five-minute decision pairs, through 2025 only.

Dependencies: numpy, pandas, pillow, pyarrow and exchange-calendars.
Build from per-month SPY minute Parquets and the repository daily CSV:
  python paired_decision_charts.py build MONTHS_DIR DAILY_CSV OUT_DIR
Render any indexed observation at 1920x1080:
  python paired_decision_charts.py render OUT_DIR '2022-12-22 09:35' chart.png
Write every compact daily and five-minute training image in a year:
  python paired_decision_charts.py images OUT_DIR 2022
"""
import argparse
import io
import json
import math
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ZONE = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]
FEATURES = COLS + ["ema8", "ema21"]
N = 42
BG = "#131722"
GRID = "#262c39"
TEXT = "#e3e9f0"
MUTED = "#a7afbf"
UP = "#26a69a"
DOWN = "#ef5350"
FAST = "#ffb300"
SLOW = "#558fff"

def indicators(data):
    for span in (8, 21):
        data[f"ema{span}"] = data.close.ewm(span=span, adjust=False).mean()
    return data

def raw_daily(filename):
    df = pd.read_csv(filename)
    df = df.loc[df.Date.between("2019-01-01", "2025-12-31")].copy()
    df.columns = df.columns.str.lower()
    df = df.set_index(pd.to_datetime(df.date)).sort_index()[COLS].astype(float)
    if not df.index.is_unique or df.index[-1] != pd.Timestamp("2025-12-31"):
        raise ValueError("Expected the repository's unadjusted pre-2026 daily history.")
    return indicators(df)

def load_intraday(months):
    frames = []
    for period in pd.period_range("2019-12", "2025-12", freq="M"):
        f = Path(months) / (str(period) + ".parquet")
        p = pd.read_parquet(f, columns=["timestamp", *COLS])
        ts = pd.to_datetime(p.pop("timestamp"), utc=True).dt.tz_convert(ZONE).dt.tz_localize(None)
        if (ts.dt.to_period("M") != period).any():
            raise ValueError(f"Wrong month in {f}")
        frames.append(p.set_axis(pd.DatetimeIndex(ts)))
    frame = pd.concat(frames).sort_index()
    if not frame.index.is_unique:
        raise ValueError("Duplicate SPY minute timestamps")
    frame = frame.between_time("04:00","19:59")
    invalid = (~np.isfinite(frame[COLS]).all(axis=1) | (frame.volume < 0) |
               (frame.low <= 0) | (frame.low > frame[["open","close"]].min(axis=1)+.000001) |
               (frame.high < frame[["open","close"]].max(axis=1)-.000001))
    if invalid.any():
        print("Invalid source minutes excluded:", int(invalid.sum()), flush=True)
        frame = frame.loc[~invalid]
    g = frame.groupby(frame.index.floor("5min"),sort=True)
    bars = g.agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    bars["minute_count"] = g.size().astype("uint8")
    bars.index.name = "bar_open_et"
    return indicators(bars)

def build(months, daily_file, dest):
    import exchange_calendars as xc
    dest = Path(dest)
    dest.mkdir(exist_ok=True,parents=True)
    daily = raw_daily(daily_file)
    bars = load_intraday(months)
    sch = xc.get_calendar("XNYS",start="2020-01-01",end="2025-12-31").schedule
    groups = bars.groupby(bars.index.normalize()).indices
    pairs, labels, audit = [], [], []
    oo,hh,ll,cc=[bars[c].to_numpy() for c in ("open","high","low","close")]
    for key, s in sch.iterrows():
        day = pd.Timestamp(key).tz_localize(None)
        op, end = [pd.Timestamp(s[k]).tz_convert(ZONE).tz_localize(None) for k in ("open","close")]
        clocks = pd.date_range(op, end, freq="5min",inclusive="left")
        did = daily.index.get_indexer([day])[0]
        available = bars.index.get_indexer(clocks)
        present = available >= 0
        reg = bars.iloc[available[present]]
        reasons = []
        if did < N: reasons.append("daily context unavailable")
        if len(reg) == 0: reasons.append("missing regular session")
        open_diff = close_diff = np.nan
        if len(reg) and did >= N:
            if present[0]: open_diff = abs(reg.open.iloc[0]/daily.open.iloc[did]-1)
            if present[-1]: close_diff = abs(reg.close.iloc[-1]/daily.close.iloc[did]-1)
            if max(np.nan_to_num(open_diff),np.nan_to_num(close_diff)) > .01:
                reasons.append("provider price basis differs by over 1%")
        positions = groups.get(day, [])
        pre = bars.iloc[positions]
        pre = pre[pre.index < op]
        audit.append(dict(session=day,expected=len(clocks),observed=len(reg),
                          observed_minutes=int(reg.minute_count.sum()),expected_minutes=5*len(clocks),
                          open_diff=open_diff,close_diff=close_diff,
                          pm_bars=len(pre),reason="; ".join(reasons)))
        if reasons: continue
        pm_hi = pre.high.max()
        pm_lo = pre.low.min()
        for j in np.flatnonzero(present):
            i = int(available[j])
            if i < N-1: continue
            decision = clocks[j] + pd.Timedelta(minutes=5)
            pairs.append((decision,did-1,i,"train" if day.year<=2024 else "validation",pm_hi,pm_lo))
            future=[]
            for horizon in (3,6,12):
                k=j+horizon
                if k>=len(clocks) or not present[j+1:k+1].all():
                    future.extend([np.nan]*3)
                    continue
                a=available[j+1]; b=available[k]
                # The next observed bar is tradable only after the decision close.
                price=oo[a]
                future.extend([cc[b]/price-1,hh[a:b+1].max()/price-1,ll[a:b+1].min()/price-1])
            labels.append([decision,*future])
    pairs=pd.DataFrame(pairs,columns=["decision_et","daily_end","intraday_end","split","pm_high","pm_low"]).set_index("decision_et")
    labels=pd.DataFrame(labels,columns=["decision_et",*[f"{k}_{m}m" for m in (15,30,60) for k in ("return","max_up","min_down")]]).set_index("decision_et")
    audit=pd.DataFrame(audit).set_index("session")
    assert pairs.index.is_unique and labels.index.equals(pairs.index) and pairs.index.max().year<=2025
    assert np.all(daily.index.values[pairs.daily_end] < pairs.index.normalize().values)
    assert np.all(bars.index.values[pairs.intraday_end]+np.timedelta64(5,'m') == pairs.index.values)
    for name,x in (("daily",daily),("five_minute",bars),("pairs",pairs),("outcomes",labels),("session_audit",audit)):
        x.to_parquet(dest/(name+".parquet"),compression="zstd")
    print("COVERAGE",pairs.groupby(pairs.index.year).size().to_dict(), flush=True)
    print("SESSION AUDIT",len(audit),"sessions, expected",audit.expected.sum(),"observed",audit.observed.sum(),
          "eligible",len(pairs),"quarantined",int(audit.reason.ne("").sum()),flush=True)
    return daily,bars,pairs,labels,audit

def font(size,bold=False):
    face="DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:return ImageFont.truetype(face,size)
    except OSError:
        from matplotlib import font_manager
        prop=font_manager.FontProperties(family="DejaVu Sans",weight="bold" if bold else "normal")
        return ImageFont.truetype(font_manager.findfont(prop),size)

def plot(d, data, bounds, rich=False, guides=()):
    x0,y0,x1,y1=bounds
    prices=data[:,[2,1,5,6]]
    lo,hi=prices.min(),prices.max()
    span=max(hi-lo,.001)
    lo-=span*.065; hi+=span*.065
    h=y1-y0
    price_bottom=y0+int(h*.76)
    vol_top=y0+int(h*.83)
    def py(v):return np.rint(y0+(hi-v)/(hi-lo)*(price_bottom-y0)).astype(int)
    def vy(v):return np.rint(y1-v/max(data[:,4].max(),1)*(y1-vol_top)).astype(int)
    px=(x1-x0)/N
    xx=np.rint(x0+(np.arange(N)+.5)*px).astype(int)
    body=max(1,int(px*.29))
    if rich:
        for v in np.linspace(lo,hi,6):
            y=int(py(v));d.line((x0,y,x1,y),fill=GRID,width=1);d.text((x1+8,y-10),f"{v:.2f}",font=font(17),fill=MUTED)
        for at in [0,10,20,30,41]:
            x=int(xx[at]);d.line((x,y0,x,y1),fill=GRID,width=1)
        for level,tag,color in guides:
            if np.isfinite(level) and lo < level < hi:
                y=int(py(level))
                for x in range(x0,x1-7,14):d.line((x,y,x+7,y),fill=color,width=1)
    oy,hy,ly,cy,fy,sy=[py(data[:,i]) for i in (0,1,2,3,5,6)]
    volumes=vy(data[:,4])
    for j,x in enumerate(xx):
        x=int(x); c=UP if data[j,3]>=data[j,0] else DOWN
        d.line((x,int(hy[j]),x,int(ly[j])),fill=c,width=2 if rich else 1)
        d.rectangle((x-body,int(min(oy[j],cy[j])),x+body,int(max(oy[j],cy[j]))),fill=c)
        d.rectangle((x-body,int(volumes[j]),x+body,y1),fill=("#227c74" if c==UP else "#ae484c"))
    for ys,color in ((fy,FAST),(sy,SLOW)):
        d.line(list(zip(xx.tolist(),ys.tolist())),fill=color,width=3 if rich else 1)
    if rich:
        d.text((x0,price_bottom+11),"VOLUME",font=font(16),fill=MUTED)

def compact(values):
    image=Image.new("RGB",(168,96),BG)
    plot(ImageDraw.Draw(image),values,(0,1,167,95))
    return image

def rich_pair(daily,bars,pairs,decision):
    decision=pd.Timestamp(decision)
    row=pairs.loc[decision]
    D=daily.iloc[int(row.daily_end)-41:int(row.daily_end)+1]
    I=bars.iloc[int(row.intraday_end)-41:int(row.intraday_end)+1]
    canvas=Image.new("RGB",(1920,1080),BG);d=ImageDraw.Draw(canvas)
    d.text((60,38),"SPY",font=font(56,True),fill=TEXT)
    d.text((198,62),"SPDR S&P 500 ETF TRUST",font=font(20),fill=MUTED)
    d.text((1855,65),decision.strftime("%d %b %Y   %H:%M ET"),font=font(27),fill=TEXT,anchor="ra")
    d.line((54,135,1860,135),fill=GRID,width=2)
    d.line((958,174,958,975),fill=GRID,width=2)
    for k,(df,title,subtitle) in enumerate([(D,"Daily context",D.index[-1].strftime("Through %d %b %Y")),
                       (I,"5-minute decision",decision.strftime("Completed at %H:%M ET"))]):
        x0=59+k*925; x1=x0+798
        d.text((x0,177),title,font=font(39,True),fill=TEXT)
        d.text((x0,238),subtitle,font=font(22),fill=MUTED)
        d.text((x0,292),f"EMA8  {df.ema8.iloc[-1]:.2f}",font=font(20,True),fill=FAST)
        d.text((x0+223,292),f"EMA21  {df.ema21.iloc[-1]:.2f}",font=font(20,True),fill=SLOW)
        r=df.iloc[-1]
        d.text((x0,338),f"O {r.open:.2f}     H {r.high:.2f}     L {r.low:.2f}     C {r.close:.2f}",font=font(19),fill=TEXT)
        guides=[] if k==0 else [(row.pm_high,"PM H","#8b76bd"),(row.pm_low,"PM L","#8b76bd"),(D.close.iloc[-1],"prior","#95a0ab")]
        plot(d,df[FEATURES].values,(x0,406,x1,898),True,guides)
        for j,anchor in [(0,'la'),(14,'ma'),(28,'ma'),(41,'ra')]:
            f="%d %b" if k==0 else "%H:%M"
            d.text((int(x0+(j+.5)*(x1-x0)/42),914),df.index[j].strftime(f),font=font(17),fill=MUTED,anchor=anchor)
    d.line((54,973,1860,973),fill=GRID,width=2)
    d.text((60,999),"42 completed candles per panel",font=font(20),fill=MUTED)
    def fmt(v):return f"{v:.2f}" if np.isfinite(v) else "unavailable"
    d.text((1855,999),f"Prior close  {D.close.iloc[-1]:.2f}     Premarket high  {fmt(row.pm_high)}     Premarket low  {fmt(row.pm_low)}",anchor="ra",font=font(20),fill=MUTED)
    return canvas

def pngbytes(im):
    f=io.BytesIO();im.save(f,format="PNG",compress_level=3);return f.getvalue()

def year_images(dest,year,daily,bars,pairs):
    dest=Path(dest)
    ids=pairs[pairs.index.year==year]
    da=daily[FEATURES].to_numpy(); ba=bars[FEATURES].to_numpy()
    path=dest/f"SPY_chart_pairs_{year}.zip"
    temp=path.with_suffix(".partial")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i in ids.daily_end.unique():
            z.writestr("daily/"+str(int(i))+".png",pngbytes(compact(da[i-41:i+1])))
        for i in ids.intraday_end:
            z.writestr("five_minute/"+str(int(i))+".png",pngbytes(compact(ba[i-41:i+1])))
        z.writestr("pairs.csv",ids.to_csv())
    temp.replace(path)
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(ids)+ids.daily_end.nunique()+1
        assert z.testzip() is None
    return path

def load_views(dest):
    return [pd.read_parquet(Path(dest)/(k+".parquet")) for k in ("daily","five_minute","pairs")]

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    b=sub.add_parser("build");b.add_argument("months");b.add_argument("daily");b.add_argument("dest")
    r=sub.add_parser("render");r.add_argument("dest");r.add_argument("decision");r.add_argument("png")
    y=sub.add_parser("images");y.add_argument("dest");y.add_argument("year",type=int)
    args=parser.parse_args()
    if args.command=="build":build(args.months,args.daily,args.dest)
    elif args.command=="images":year_images(args.dest,args.year,*load_views(args.dest))
    else:rich_pair(*load_views(args.dest),args.decision).save(args.png)
