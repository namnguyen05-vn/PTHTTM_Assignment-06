"""Causal sequence construction and chronological split audit."""
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT, DATA
from src.io import write_json, sha256, read_json
import numpy as np
import pandas as pd

def read_source(name, member):
    path = DATA/"raw"/(name+".download")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entry = next(n for n in archive.namelist() if Path(n).name == member)
            with archive.open(entry) as stream:
                return pd.read_csv(stream)
    return pd.read_csv(path)

def boundaries(times):
    days = np.unique(times // 86400)
    return int(days[int(len(days)*.70)]*86400), int(days[int(len(days)*.85)]*86400)

def group_bounds(new_group):
    starts = np.maximum.accumulate(np.where(new_group, np.arange(len(new_group)), 0))
    changes = np.flatnonzero(new_group)
    ends = np.repeat(np.r_[changes[1:]-1, len(new_group)-1], np.diff(np.r_[changes, len(new_group)]))
    return starts, ends

def persist(name, features, feature_names, endpoints, lengths, labels, times, groups,
            cutoffs, details, extras, feature_time):
    folder = DATA/"prepared"/name
    folder.mkdir(parents=True, exist_ok=True)
    splits = np.where(times < cutoffs[0], 0, np.where(times < cutoffs[1], 1, 2)).astype("uint8")
    fit = features[feature_time < cutoffs[0]].astype("float64")
    mean, scale = fit.mean(0), fit.std(0)
    scale[scale < 1e-8] = 1.0
    normalized = ((features-mean)/scale).astype("float32")
    arrays = dict(features=normalized, endpoints=endpoints.astype("int64"),
                  lengths=lengths.astype("int32"), labels=labels, times=times.astype("int64"),
                  groups=groups, split=splits, **extras)
    for key, value in arrays.items():
        np.save(folder/(key+".npy"), value, allow_pickle=False)
    lengths_config = read_json(ROOT/"configs/main.json")["sequence_lengths"]
    metadata = dict(dataset=name, features=feature_names, input_dim=features.shape[1],
                    sequence_length=lengths_config[name], cutoffs_utc=[pd.Timestamp(t, unit="s", tz="UTC").isoformat() for t in cutoffs],
                    split_sizes={k:int((splits==i).sum()) for i,k in enumerate(["train","val","test"])},
                    scaler_mean=mean.tolist(), scaler_scale=scale.tolist(),
                    raw_sha256=sha256(DATA/"raw"/(name+".download")), **details)
    if name == "retailrocket":
        metadata["class_counts"] = {k:np.bincount(labels[splits==i], minlength=3).tolist() for i,k in enumerate(["train","val","test"])}
        metadata["labels"] = ["view","addtocart","transaction"]
        metadata["target_scale"] = 1.0
    else:
        metadata["target_scale"] = float(max(labels[splits==0].std(),1e-6))
        metadata["target_summary"] = {k:dict(mean=float(labels[splits==i].mean()), std=float(labels[splits==i].std())) for i,k in enumerate(["train","val","test"])}
    assert all(n > 0 for n in metadata["split_sizes"].values())
    write_json(folder/"metadata.json", metadata)
    write_json(ROOT/"results"/(name+"_audit.json"), metadata)
    print(name, metadata["split_sizes"], flush=True)

def retailrocket():
    frame = read_source("retailrocket","events.csv")
    original = len(frame)
    frame = frame.drop_duplicates()
    exact_duplicates = original-len(frame)
    ambiguous = frame.duplicated(["visitorid","timestamp"], keep=False)
    ties = int(ambiguous.sum())
    frame = frame.loc[~ambiguous].sort_values(["visitorid","timestamp"],kind="stable").reset_index(drop=True)
    assert frame[["timestamp","visitorid","event","itemid"]].notna().all().all()
    code = frame.event.map(dict(view=0,addtocart=1,transaction=2))
    assert code.notna().all()
    event = code.to_numpy(dtype="int64")
    milliseconds = frame.timestamp.to_numpy(dtype="int64")
    times = milliseconds//1000
    visitors = frame.visitorid.to_numpy(dtype="int64")
    item = frame.itemid.to_numpy(dtype="int64")
    gaps = np.r_[0, np.diff(milliseconds)/1000.]
    new = np.r_[True, visitors[1:]!=visitors[:-1]] | (gaps>1800)
    starts, ends = group_bounds(new)
    gaps[new] = 0
    position = np.arange(len(frame))-starts
    previous_same = np.r_[False,item[1:]==item[:-1]] & ~new
    dt = pd.to_datetime(times, unit="s", utc=True)
    hour = dt.hour.to_numpy()+dt.minute.to_numpy()/60
    dow = dt.dayofweek.to_numpy()
    features = np.column_stack([np.eye(3)[event], np.log1p(gaps), previous_same,
        np.log1p(position), np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]).astype("float32")
    cuts = boundaries(times)
    segment_split_start = np.searchsorted(cuts, times[starts],side="right")
    segment_split_end = np.searchsorted(cuts, times[ends],side="right")
    eligible = (position>=1) & (segment_split_start==segment_split_end)
    target = np.flatnonzero(eligible)
    length = read_json(ROOT/"configs/main.json")["sequence_lengths"]["retailrocket"]
    counts = np.bincount(event, minlength=3)
    details = dict(raw_rows=original, exact_duplicates_removed=exact_duplicates,
        ambiguous_timestamp_rows_removed=ties, retained_events=len(frame),
        raw_visitors=int(frame.visitorid.nunique()), sessions=int(new.sum()),
        singleton_sessions=int(np.sum(starts==ends)), crossing_boundary_target_rows_removed=int(((position>=1)&~eligible).sum()),
        event_counts=counts.tolist(), date_min=str(dt.min()), date_max=str(dt.max()),
        task="Next event type, conditional on another event existing within the session.",
        session_gap_seconds=1800, timestamp_unit_source="milliseconds",
        split_policy="Global dates; sessions crossing boundaries excluded; returning visitors allowed.",
        excluded_columns=["transactionid (outcome identifier)","item properties (not required)"])
    persist("retailrocket", features, ["view","addtocart","transaction","log_gap_seconds","same_item_as_previous",
        "log_session_position","hour_sin","hour_cos","weekday_sin","weekday_cos"],
        target-1, np.minimum(position[target],length), event[target],times[target],visitors[target],
        cuts,details,dict(previous_event=event[target-1],session=np.cumsum(new)[target],
                          endpoint_time=times[target-1]),times)

def sp500():
    frame=read_source("sp500","all_stocks_5yr.csv")
    original=len(frame)
    frame["date"]=pd.to_datetime(frame.date,errors="coerce")
    frame=frame.drop_duplicates()
    duplicates=original-len(frame)
    conflict=frame.duplicated(["Name","date"],keep=False)
    conflicts=int(conflict.sum())
    numeric=["open","high","low","close","volume"]
    for col in numeric:
        frame[col]=pd.to_numeric(frame[col],errors="coerce")
    values=frame[numeric].to_numpy()
    valid=np.isfinite(values).all(1)&frame.date.notna().to_numpy()&frame.Name.notna().to_numpy()
    valid&=(values[:,:4]>0).all(1)&(values[:,4]>=0)
    valid&=(frame.high>=frame[["open","close","low"]].max(axis=1)-1e-6).to_numpy()
    valid&=(frame.low<=frame[["open","close","high"]].min(axis=1)+1e-6).to_numpy()
    removed=int((~valid&~conflict).sum())
    frame=frame.loc[valid&~conflict].sort_values(["Name","date"]).reset_index(drop=True)
    tickers=sorted(frame.Name.unique().tolist())
    codes=frame.Name.map({v:i for i,v in enumerate(tickers)}).to_numpy(dtype="int32")
    times=(frame.date.astype("int64")//10**9).to_numpy()
    new=np.r_[True,codes[1:]!=codes[:-1]] | (np.r_[0,np.diff(times)]>10*86400)
    starts,_=group_bounds(new)
    close=frame.close.to_numpy(dtype="float64")
    opening=frame.open.to_numpy(dtype="float64")
    previous=np.r_[close[0],close[:-1]]
    log_return=np.log(close/previous)
    gap=np.log(opening/previous)
    log_volume=np.log1p(frame.volume.to_numpy())
    volume_change=np.r_[0,np.diff(log_volume)]
    log_return[new]=0;gap[new]=0;volume_change[new]=0
    dow=frame.date.dt.dayofweek.to_numpy()
    features=np.column_stack([log_return,np.log(close/opening),
        np.log(frame.high.to_numpy()/frame.low.to_numpy()),gap,volume_change,
        np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]).astype("float32")
    length=read_json(ROOT/"configs/main.json")["sequence_lengths"]["sp500"]
    target=np.flatnonzero(np.arange(len(frame))-starts>=length+1)
    cuts=boundaries(times)
    details=dict(raw_rows=original,exact_duplicates_removed=duplicates,
        conflicting_date_rows_removed=conflicts,invalid_rows_removed=removed,
        retained_rows=len(frame),tickers=tickers,ticker_count=len(tickers),
        date_min=str(frame.date.min()),date_max=str(frame.date.max()),
        extreme_target_returns_above_50pct=int((np.abs(log_return[target])>np.log(1.5)).sum()),
        task="One observed trading session ahead log return; implied closing price is also evaluated.",
        split_policy="Global target dates; preceding historical context can cross a split boundary.",
        adjustment_status="The Kaggle card does not establish a point-in-time corporate-action adjustment policy.",
        gap_reset_calendar_days=10)
    persist("sp500",features,["close_log_return","intraday_log_return","log_high_low_range",
        "opening_log_gap","log_volume_change","weekday_sin","weekday_cos"],
        target-1,np.full(len(target),length),log_return[target].astype("float32"),
        times[target],codes[target],cuts,details,
        dict(base_close=previous[target],target_close=close[target],endpoint_time=times[target-1]),times)

if __name__=="__main__":
    retailrocket()
    sp500()
