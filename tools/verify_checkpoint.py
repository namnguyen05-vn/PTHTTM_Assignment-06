"""Strict reload of one full checkpoint in a fresh backend process."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import read_json,write_json
from src.data import load_data,make_batch
from src.training import checkpoint_prefix
from src.backends import create_adapter
import numpy as np
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("dataset",choices=["retailrocket","sp500"])
    p.add_argument("framework",choices=["pytorch","keras"])
    args=p.parse_args()
    folder=ROOT/"results/runs"/(args.dataset+"_"+args.framework+"_seed42")
    config=read_json(folder/"config.json")
    runtime=read_json(folder/"runtime.json")
    data=load_data(args.dataset)
    adapter=create_adapter(args.framework,config,data["metadata"],cpu=runtime["device"]=="cpu")
    if adapter.device!=runtime["device"]:
        raise RuntimeError("Strict reload requires the recorded device family.")
    state=read_json(folder/"state.json")
    adapter.restore(checkpoint_prefix(folder,state["best_epoch"]))
    with np.load(folder/"outputs.npz") as z:
        ids=z["ids"][:config["batch_size"]];expected=z["output"][:len(ids)]
    x,lengths,_=make_batch(data,ids)
    actual=adapter.predict(x,lengths)
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
    record=dict(dataset=args.dataset,framework=args.framework,samples=len(ids),
        max_abs_error=float(np.abs(actual-expected).max()),device=adapter.device,passed=True)
    write_json(ROOT/"results"/("checkpoint_"+args.dataset+"_"+args.framework+".json"),record)
    print(record)
