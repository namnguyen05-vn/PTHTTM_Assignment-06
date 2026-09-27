"""Independent structural checks of every prepared sample."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT,DATA
from src.io import write_json,read_json,sha256
from src.data import load_data,split_ids,make_batch
from src.teaching import last_state_gradient,finite_difference
import numpy as np

if __name__=="__main__":
    records=[]
    for name in ["retailrocket","sp500"]:
        d=load_data(name);m=d["metadata"]
        assert np.isfinite(d["features"]).all() and np.isfinite(d["labels"]).all()
        assert len(d["endpoints"])==len(d["labels"])==sum(m["split_sizes"].values())
        assert np.all(d["lengths"]>=1) and np.all(d["lengths"]<=m["sequence_length"])
        assert np.all(d["endpoints"]-d["lengths"]+1>=0)
        assert np.all(d["endpoint_time"]<=d["times"])
        for earlier,later in [("train","val"),("val","test")]:
            assert d["times"][split_ids(d,earlier)].max()<d["times"][split_ids(d,later)].min()
        if name=="retailrocket":
            sets=[set(d["session"][split_ids(d,s)]) for s in ["train","val","test"]]
            assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        rng=np.random.default_rng(101)
        ids=rng.choice(len(d["labels"]),size=101,replace=False)
        x,lengths,y=make_batch(d,ids)
        for i,(sample,n) in enumerate(zip(ids,lengths)):
            np.testing.assert_array_equal(x[i,n:],0)
            np.testing.assert_array_equal(x[i,n-1],d["features"][d["endpoints"][sample]])
        assert m["raw_sha256"]==sha256(DATA/"raw"/(name+".download"))
        records.append(dict(dataset=name,samples=len(d["labels"]),chronology_passed=True,
            finite_arrays=True,padding_samples_checked=len(ids),passed=True))
    rng=np.random.default_rng(2)
    x=rng.normal(size=(3,2));wx=rng.normal(size=(2,2));wh=rng.normal(size=(2,2));b=np.zeros(2);target=np.ones(2)
    _,analytic=last_state_gradient(x,wx,wh,b,target)
    numerical=[
        finite_difference(lambda a:last_state_gradient(x,a,wh,b,target)[0],wx),
        finite_difference(lambda a:last_state_gradient(x,wx,a,b,target)[0],wh),
        finite_difference(lambda a:last_state_gradient(x,wx,wh,a,target)[0],b)]
    for a,n in zip(analytic,numerical):
        np.testing.assert_allclose(a,n,atol=1e-7,rtol=1e-6)
    write_json(ROOT/"results/data_verification.json",dict(datasets=records,numpy_bptt_gradient_passed=True))
    print("Data chronology, padding, source hashes and BPTT gradients passed.")
