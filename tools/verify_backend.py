"""One backend per process: algebra, masking, checkpoint and optimizer tests."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import read_json,write_json
from src.training import configuration,train
from src.backends import create_adapter
from src.initialization import initial_weights
from src.teaching import rnn_forward
import numpy as np

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("framework",choices=["pytorch","keras"])
    args=parser.parse_args()
    config,data=configuration("retailrocket",args.framework,smoke=True)
    adapter=create_adapter(args.framework,config,data["metadata"],cpu=True)
    rng=np.random.default_rng(19)
    x=rng.normal(size=(5,20,10)).astype("float32")
    lengths=np.array([20,12,4,1,17],dtype="int32")
    w=initial_weights(10,32,3,42)
    reference=np.array([rnn_forward(a[:n],w["wx"],w["wh"],w["bh"])[-1]@w["wy"]+w["by"] for a,n in zip(x,lengths)])
    actual=adapter.predict(x,lengths)
    np.testing.assert_allclose(actual,reference,atol=2e-6,rtol=2e-6)
    changed=x.copy()
    for row,n in enumerate(lengths):
        changed[row,n:]=10000.
    np.testing.assert_allclose(adapter.predict(changed,lengths),actual,atol=1e-6,rtol=1e-6)
    prefix=ROOT/"results/verification"/(args.framework+"_roundtrip")
    prefix.parent.mkdir(parents=True,exist_ok=True)
    y=np.array([0,1,2,0,1],dtype="int64")
    adapter.train_batch(x,lengths,y)
    adapter.save(prefix)
    adapter.train_batch(x,lengths,y)
    continued=adapter.predict(x,lengths)
    adapter.restore(prefix)
    adapter.train_batch(x,lengths,y)
    resumed=adapter.predict(x,lengths)
    np.testing.assert_allclose(resumed,continued,atol=1e-6,rtol=1e-6)
    record=dict(framework=args.framework,device=adapter.device,version=adapter.version,
        initial_numpy_max_error=float(np.max(np.abs(actual-reference))),
        padded_values_ignored=True,optimizer_resume_max_error=float(np.max(np.abs(continued-resumed))),
        trainable_parameters=adapter.trainable,passed=True)
    write_json(ROOT/"results"/("verification_"+args.framework+".json"),record)
    print(record)
