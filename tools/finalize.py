"""Post-training verification and report build. This script never trains an RNN."""
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.io import write_json
if __name__=="__main__":
    expected=[ROOT/"results/runs"/f"{d}_{f}_seed42"/"metrics.json"
        for d in ["retailrocket","sp500"] for f in ["pytorch","keras"]]
    if not all(p.exists() for p in expected):
        raise SystemExit("Four complete runs are required; train with RUN_TRAINING.bat first.")
    steps=[["tools/verify_results.py"]]
    steps += [["tools/verify_checkpoint.py",d,f]
        for d in ["retailrocket","sp500"] for f in ["pytorch","keras"]]
    steps += [["tools/baselines.py"],["tools/figures.py","--results"],
        ["tools/execute_notebooks.py"],["report/build_report.py"]]
    status=dict(status="running",finished=[])
    for step in steps:
        status["stage"]=" ".join(step)
        write_json(ROOT/"results/build_status.json",status)
        subprocess.run([sys.executable]+step,cwd=ROOT,check=True)
        status["finished"].append(status["stage"])
    status.update(status="ready_for_visual_review")
    write_json(ROOT/"results/build_status.json",status)
