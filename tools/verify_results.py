"""Verify full runs only; never turns smoke output into final evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import read_json,write_json
from src.data import load_data,split_ids
from src.evaluation import evaluate
import numpy as np

if __name__=="__main__":
    records=[]
    for name in ["retailrocket","sp500"]:
        data=load_data(name);ids=split_ids(data,"test")
        for framework in ["pytorch","keras"]:
            folder=ROOT/"results/runs"/(name+"_"+framework+"_seed42")
            m=read_json(folder/"metrics.json")
            assert not m["smoke"] and m["evaluation_split"]=="test"
            with np.load(folder/"outputs.npz") as z:
                np.testing.assert_array_equal(z["ids"],ids)
                np.testing.assert_array_equal(z["labels"],data["labels"][ids])
                expected=evaluate(data,ids,z["output"])
            for key,value in expected.items():
                if isinstance(value,(int,float)):
                    assert np.isclose(m[key],value,atol=1e-8,rtol=1e-7),(name,framework,key)
                else:
                    assert m[key]==value,(name,framework,key)
            state=read_json(folder/"state.json")
            assert m["best_epoch"]==state["best_epoch"]
            records.append(dict(dataset=name,framework=framework,samples=len(ids),verified=True))
    write_json(ROOT/"results/metric_verification.json",records)
    print("Verified four complete RNN test results.")
