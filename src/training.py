"""Epoch-level restart with a committed checkpoint pointer and separate smoke results."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from filelock import FileLock
from .paths import ROOT, DATA
from .io import read_json,write_json,save_npz,sha256
from .data import load_data,split_ids,batches
from .backends import create_adapter
from .evaluation import evaluate

def infer(adapter,data,ids,batch_size):
    return np.concatenate([adapter.predict(x,lengths)
        for _,(x,lengths,_) in batches(data,ids,batch_size)],axis=0)

def configuration(dataset,framework,smoke=False):
    config=read_json(ROOT/"configs/main.json")
    data=load_data(dataset)
    config={k:v for k,v in config.items() if k not in ["datasets","frameworks","sequence_lengths"]}
    if smoke:
        config.update(max_epochs=2,patience=2,batch_size=64)
    counts=np.bincount(data["labels"][split_ids(data,"train")].astype("int64"),minlength=3) if dataset=="retailrocket" else np.ones(3)
    weights=np.sqrt(counts.sum()/(3*np.maximum(counts,1)))
    weights/=np.dot(counts,weights)/counts.sum()
    config.update(dataset=dataset,framework=framework,smoke=smoke,
        class_weights=weights.tolist(),
        metadata_sha256=sha256(DATA/"prepared"/dataset/"metadata.json"),
        selection="validation macro-F1" if dataset=="retailrocket" else "validation RMSE")
    return config,data

def checkpoint_prefix(folder,epoch):
    return folder/"checkpoints"/("epoch_%04d"%epoch)

def train(dataset,framework,smoke=False,cpu=False,stop_after=None,run_tag=None):
    config,data=configuration(dataset,framework,smoke)
    name=dataset+"_"+framework+"_seed"+str(config["seed"])
    base=ROOT/"results"/("smoke" if smoke else "runs")
    folder=base/(run_tag or name)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/"run.lock"),timeout=0):
        if (folder/"config.json").exists():
            previous=read_json(folder/"config.json")
            if previous!=config:
                raise RuntimeError("Configuration changed. Use a new result directory; do not mix runs.")
        else:
            write_json(folder/"config.json",config)
        if (folder/"metrics.json").exists():
            print("Already completed:",name,flush=True)
            return 0
        return run_training(folder,config,data,cpu,stop_after)

def run_training(folder,config,data,cpu,stop_after):
    adapter=create_adapter(config["framework"],config,data["metadata"],cpu)
    write_json(folder/"runtime.json",dict(framework=adapter.version,device=adapter.device,
        trainable_parameters=adapter.trainable))
    train_ids=split_ids(data,"train");val_ids=split_ids(data,"val")
    if config["smoke"]:
        rng=np.random.default_rng(config["seed"])
        train_ids=np.sort(rng.choice(train_ids,min(512,len(train_ids)),replace=False))
        val_ids=np.sort(rng.choice(val_ids,min(128,len(val_ids)),replace=False))
    (folder/"checkpoints").mkdir(exist_ok=True)
    state_path=folder/"state.json"
    if state_path.exists():
        state=read_json(state_path)
        adapter.restore(checkpoint_prefix(folder,state["epoch"]))
        print("Resuming after epoch",state["epoch"],flush=True)
    else:
        state=dict(epoch=0,best_epoch=0,best_score=None,wait=0,history=[])
    classification=config["dataset"]=="retailrocket"
    for epoch in range(state["epoch"]+1,config["max_epochs"]+1):
        if state["wait"]>=config["patience"]:
            break
        if (ROOT/"PAUSE_TRAINING").exists():
            print("Paused before next epoch.",flush=True)
            return 75
        start=time.perf_counter();total=0.
        for _,(x,lengths,y) in batches(data,train_ids,config["batch_size"],config["seed"]+epoch):
            if not classification:
                y=y/data["metadata"]["target_scale"]
            total+=adapter.train_batch(x,lengths,y)*len(y)
        output=infer(adapter,data,val_ids,config["batch_size"])
        metrics=evaluate(data,val_ids,output)
        score=metrics["macro_f1"] if classification else -metrics["rmse"]
        improved=state["best_score"] is None or score>state["best_score"]+config["min_delta"]
        row=dict(epoch=epoch,train_loss=total/len(train_ids),
                 val_score=score,val_loss=metrics["log_loss"] if classification else metrics["rmse"],
                 seconds=time.perf_counter()-start)
        adapter.save(checkpoint_prefix(folder,epoch))
        state["history"].append(row)
        state.update(epoch=epoch,wait=0 if improved else state["wait"]+1)
        if improved:
            state.update(best_epoch=epoch,best_score=score)
        # Atomic JSON publishes a checkpoint only after all its files are written.
        write_json(state_path,state)
        pd.DataFrame(state["history"]).to_csv(folder/"history.csv",index=False)
        print(config["dataset"],config["framework"],"epoch",epoch,"train",round(row["train_loss"],6),
              "validation",round(row["val_loss"],6),flush=True)
        keep={epoch,state["best_epoch"]}
        for candidate in (folder/"checkpoints").glob("epoch_*"):
            number=int(candidate.name.split(".")[0].split("_")[1])
            if number not in keep:
                candidate.unlink()
        if stop_after is not None and epoch>=stop_after:
            print("Stopped at a saved epoch for verification.",flush=True)
            return 75
        if (ROOT/"PAUSE_TRAINING").exists():
            print("Epoch saved. Safe to close this process and sleep.",flush=True)
            return 75
    adapter.restore(checkpoint_prefix(folder,state["best_epoch"]))
    test_ids=val_ids if config["smoke"] else split_ids(data,"test")
    output=infer(adapter,data,test_ids,config["batch_size"])
    metrics=evaluate(data,test_ids,output)
    save_npz(folder/"outputs.npz",ids=test_ids,labels=np.asarray(data["labels"][test_ids]),
             output=output,timestamps=np.asarray(data["times"][test_ids]))
    prediction=output.argmax(1) if classification else output[:,0]*data["metadata"]["target_scale"]
    table=pd.DataFrame(dict(sample_id=test_ids,timestamp=data["times"][test_ids],
        group=data["groups"][test_ids],actual=data["labels"][test_ids],prediction=prediction))
    if not classification:
        table["actual_close"]=data["target_close"][test_ids]
        table["predicted_close"]=data["base_close"][test_ids]*np.exp(np.clip(prediction,-20,20))
    table.to_csv(folder/"predictions.csv",index=False)
    metrics.update(dataset=config["dataset"],framework=config["framework"],seed=config["seed"],
        best_epoch=state["best_epoch"],completed_epochs=state["epoch"],
        train_seconds=sum(r["seconds"] for r in state["history"]),samples=len(test_ids),
        trainable_parameters=adapter.trainable,smoke=config["smoke"],
        evaluation_split="validation smoke subset" if config["smoke"] else "test")
    # Completion marker is last, never written for an interrupted run.
    write_json(folder/"metrics.json",metrics)
    print("Completed",folder.name,flush=True)
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["retailrocket","sp500"],required=True)
    parser.add_argument("--framework",choices=["pytorch","keras"],required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--run-tag")
    args=parser.parse_args()
    try:
        raise SystemExit(train(**vars(args)))
    except KeyboardInterrupt:
        print("\nInterrupted. The next run restores the last fully saved epoch.",flush=True)
        raise SystemExit(130)
