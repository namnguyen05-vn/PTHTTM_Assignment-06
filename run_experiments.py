"""Sequential four-run queue; full training starts only by an explicit invocation."""
import argparse
import os
import subprocess
import sys
from src.paths import ROOT
from filelock import FileLock

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dataset",choices=["retailrocket","sp500"])
    parser.add_argument("--framework",choices=["pytorch","keras"])
    args=parser.parse_args()
    (ROOT/"runtime").mkdir(exist_ok=True)
    with FileLock(str(ROOT/"runtime/queue.lock"),timeout=0):
        if args.resume and (ROOT/"PAUSE_TRAINING").exists():
            (ROOT/"PAUSE_TRAINING").unlink()
        for dataset in ([args.dataset] if args.dataset else ["retailrocket","sp500"]):
            for framework in ([args.framework] if args.framework else ["pytorch","keras"]):
                if (ROOT/"PAUSE_TRAINING").exists():
                    print("Queue paused. Restart with --resume after waking the computer.")
                    raise SystemExit(75)
                command=[sys.executable,"-m","src.training","--dataset",dataset,"--framework",framework]
                if args.smoke:command.append("--smoke")
                if args.cpu:command.append("--cpu")
                code=subprocess.call(command,cwd=ROOT)
                if code:
                    raise SystemExit(code)
