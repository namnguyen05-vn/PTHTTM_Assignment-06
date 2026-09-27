"""Verify the real epoch runner across a process restart, using smoke data only."""
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import write_json
import numpy as np

if __name__=="__main__":
    records=[]
    for framework in ["pytorch","keras"]:
        tag="resume_pipeline_"+framework
        base=[sys.executable,"-m","src.training","--dataset","retailrocket",
              "--framework",framework,"--smoke","--run-tag",tag]
        result=subprocess.run(base+["--stop-after","1"],cwd=ROOT)
        if result.returncode not in [0,75]:
            raise RuntimeError("Interrupted checkpoint test failed")
        subprocess.run(base,cwd=ROOT,check=True)
        reference=ROOT/"results/smoke"/("retailrocket_"+framework+"_seed42")/"outputs.npz"
        resumed=ROOT/"results/smoke"/tag/"outputs.npz"
        with np.load(reference) as first,np.load(resumed) as second:
            np.testing.assert_array_equal(first["ids"],second["ids"])
            np.testing.assert_allclose(first["output"],second["output"],atol=1e-6,rtol=1e-6)
            error=float(np.abs(first["output"]-second["output"]).max())
        records.append(dict(framework=framework,process_restart=True,max_output_error=error,passed=True))
    write_json(ROOT/"results/resume_verification.json",records)
    print(records)
