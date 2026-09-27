"""Train-only baselines evaluated once the full experiment is complete."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.data import load_data,split_ids
from src.io import write_json,save_npz
from src.evaluation import classification_metrics,regression_metrics
import numpy as np

def run():
    records=[]
    d=load_data("retailrocket");train=split_ids(d,"train");test=split_ids(d,"test")
    frequencies=np.bincount(d["labels"][train],minlength=3)+1.
    probability=frequencies/frequencies.sum()
    counts=np.ones((3,3),dtype="float64")
    np.add.at(counts,(d["previous_event"][train],d["labels"][train]),1)
    transition=counts/counts.sum(1,keepdims=True)
    for name,probs in [("prior",np.tile(probability,(len(test),1))),("markov",transition[d["previous_event"][test]])]:
        logits=np.log(probs)
        metrics=classification_metrics(d["labels"][test],logits)
        metrics.update(dataset="retailrocket",model=name)
        records.append(metrics)
        save_npz(ROOT/"results"/("baseline_"+name+".npz"),ids=test,output=logits)
    write_json(ROOT/"results/baseline_retail_fit.json",dict(prior=probability.tolist(),transition=transition.tolist()))
    d=load_data("sp500");train=split_ids(d,"train");test=split_ids(d,"test")
    for name,value in [("zero_return",0.),("train_mean_return",float(d["labels"][train].mean()))]:
        predicted=np.full(len(test),value)
        metrics=regression_metrics(d["labels"][test],predicted,d["base_close"][test],d["target_close"][test])
        metrics.update(dataset="sp500",model=name)
        records.append(metrics)
        save_npz(ROOT/"results"/("baseline_"+name+".npz"),ids=test,prediction=predicted)
    write_json(ROOT/"results/baselines.json",records)
    return records

if __name__=="__main__":
    run()
