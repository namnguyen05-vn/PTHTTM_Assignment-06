"""Read-only arrays; windows constructed on demand instead of materializing N*T*D."""
import numpy as np
from .paths import DATA
from .io import read_json

def load_data(name):
    directory=DATA/"prepared"/name
    data={p.stem:np.load(p,mmap_mode="r",allow_pickle=False) for p in directory.glob("*.npy")}
    data["metadata"]=read_json(directory/"metadata.json")
    return data

def split_ids(data, split):
    code={"train":0,"val":1,"test":2}[split]
    return np.flatnonzero(data["split"]==code)

def make_batch(data, ids):
    ids=np.asarray(ids,dtype="int64")
    lengths=np.asarray(data["lengths"][ids],dtype="int32")
    total=data["metadata"]["sequence_length"]
    offsets=np.arange(total)[None,:]
    start=data["endpoints"][ids]-lengths+1
    positions=start[:,None]+offsets
    mask=offsets<lengths[:,None]
    positions=np.where(mask,positions,0)
    x=np.array(data["features"][positions],dtype="float32")
    x[~mask]=0.0
    y=np.array(data["labels"][ids])
    return x,lengths,y

def batches(data, ids, batch_size, seed=None):
    order=np.array(ids,copy=True)
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0,len(order),batch_size):
        selected=order[start:start+batch_size]
        yield selected,make_batch(data,selected)
