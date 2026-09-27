"""Read saved full-test outputs; produce diagnostics without model training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT, DATA
from src.io import read_json,write_json,sha256
from src.data import load_data,split_ids
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    rows=[]; paired={}; checks=[]
    for dataset in ["retailrocket","sp500"]:
        data=load_data(dataset); ids=split_ids(data,"test"); outputs={}
        for framework in ["pytorch","keras"]:
            folder=ROOT/"results/runs"/f"{dataset}_{framework}_seed42"
            metrics=read_json(folder/"metrics.json"); config=read_json(folder/"config.json")
            state=read_json(folder/"state.json")
            assert config["metadata_sha256"]==sha256(DATA/"prepared"/dataset/"metadata.json")
            history=pd.read_csv(folder/"history.csv")
            assert history.epoch.tolist()==list(range(1,state["epoch"]+1))
            np.testing.assert_allclose(history.seconds.sum(),metrics["train_seconds"],rtol=1e-10)
            best=None;best_epoch=0;wait=0
            for item in history.itertuples():
                if best is None or item.val_score>best+config["min_delta"]:
                    best=item.val_score;best_epoch=item.epoch;wait=0
                else: wait+=1
            assert best_epoch==metrics["best_epoch"]==state["best_epoch"]
            assert wait==state["wait"] and (wait>=config["patience"] or state["epoch"]==config["max_epochs"])
            with np.load(folder/"outputs.npz") as saved:
                np.testing.assert_array_equal(saved["ids"],ids)
                np.testing.assert_array_equal(saved["timestamps"],data["times"][ids])
                outputs[framework]=saved["output"].copy()
            frame=pd.read_csv(folder/"predictions.csv")
            np.testing.assert_array_equal(frame.sample_id,ids)
            np.testing.assert_array_equal(frame.group,data["groups"][ids])
            np.testing.assert_array_equal(frame.timestamp,data["times"][ids])
            np.testing.assert_allclose(frame.actual,data["labels"][ids],atol=1e-8)
            pred=outputs[framework].argmax(1) if dataset=="retailrocket" else outputs[framework][:,0]*data["metadata"]["target_scale"]
            np.testing.assert_allclose(frame.prediction,pred,atol=1e-8)
            if dataset=="sp500":
                np.testing.assert_allclose(frame.actual_close,data["target_close"][ids],rtol=1e-7)
                price=data["base_close"][ids]*np.exp(np.clip(pred,-20,20))
                np.testing.assert_allclose(frame.predicted_close,price,rtol=1e-6)
            row={k:v for k,v in metrics.items() if isinstance(v,(str,float,int,bool))}
            rows.append(row)
            checks.append(dict(dataset=dataset,framework=framework,metadata_matches=True,
                history_and_selection_verified=True,csv_predictions_verified=True))
        a,b=outputs["pytorch"],outputs["keras"]
        if dataset=="retailrocket":
            a,b=softmax(a,axis=1),softmax(b,axis=1)
            paired[dataset]=dict(max_probability_difference=float(abs(a-b).max()),
                mean_probability_difference=float(abs(a-b).mean()),
                prediction_agreement=float(np.mean(a.argmax(1)==b.argmax(1))))
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            with np.load(ROOT/"results/baseline_markov.npz") as z:
                markov=softmax(z["output"],axis=1)
            for label,ax in zip([1,2],axes):
                for name,probs in [("PyTorch",a),("Keras",b),("Markov",markov)]:
                    precision,recall,_=precision_recall_curve(data["labels"][ids]==label,probs[:,label])
                    ax.plot(recall,precision,label=name,alpha=.85)
                ax.axhline(np.mean(data["labels"][ids]==label),color="gray",ls=":",label="Tỷ lệ lớp test")
                ax.set(xlabel="Recall",ylabel="Precision",title=data["metadata"]["labels"][label],xlim=(0,1),ylim=(0,1))
                ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(ROOT/"figures/retail_pr.png",dpi=180);plt.close(fig)
        else:
            a=a[:,0]*data["metadata"]["target_scale"];b=b[:,0]*data["metadata"]["target_scale"]
            y=np.asarray(data["labels"][ids])
            paired[dataset]=dict(max_return_difference=float(abs(a-b).max()),
                mean_return_difference=float(abs(a-b).mean()),
                actual_return_std=float(y.std()),pytorch_return_std=float(a.std()),keras_return_std=float(b.std()),
                upward_share=float(np.mean(y>0)),downward_share=float(np.mean(y<0)),zero_share=float(np.mean(y==0)))
            month=pd.to_datetime(data["times"][ids],unit="s").strftime("%Y-%m")
            mean=float(data["labels"][split_ids(data,"train")].mean())
            monthly=[]
            for name,pred in [("pytorch",a),("keras",b),("zero_return",np.zeros(len(y))),("train_mean_return",np.full(len(y),mean))]:
                frame=pd.DataFrame(dict(month=month,error=pred-y))
                for group,values in frame.groupby("month"):
                    monthly.append(dict(model=name,month=group,n=len(values),rmse=float(np.sqrt(np.mean(values.error**2)))))
            pd.DataFrame(monthly).to_csv(ROOT/"results/stock_monthly_metrics.csv",index=False)
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            sample=np.random.default_rng(42).choice(len(y),min(6000,len(y)),replace=False)
            axes[0].scatter(y[sample],a[sample],s=5,alpha=.2)
            axes[0].plot([-.08,.08],[-.08,.08],color="gray",ls="--")
            axes[0].set(xlabel="Log return thực",ylabel="Log return dự đoán (PyTorch)",xlim=(-.08,.08),ylim=(-.08,.08))
            for name,pred in [("PyTorch",a),("Keras",b),("Zero return",np.zeros(len(y)))]:
                axes[1].hist(pred-y,bins=90,range=(-.06,.06),histtype="step",density=True,label=name)
            axes[1].set(xlabel="Sai số return trong khung ±0,06",ylabel="Mật độ trong khung");axes[1].legend()
            fig.tight_layout();fig.savefig(ROOT/"figures/stock_diagnostics.png",dpi=180);plt.close(fig)
            fig,ax=plt.subplots(figsize=(10,3.5))
            monthly=pd.DataFrame(monthly)
            for name,group in monthly.groupby("model"):
                ax.plot(group.month,group.rmse,marker="o",label=name)
            ax.set(xlabel="Tháng test",ylabel="RMSE log return");ax.legend(fontsize=8);ax.tick_params(axis="x",rotation=30)
            fig.tight_layout();fig.savefig(ROOT/"figures/stock_monthly.png",dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/"results/experiment_summary.csv",index=False)
    write_json(ROOT/"results/paired_framework_analysis.json",paired)
    write_json(ROOT/"results/provenance_verification.json",checks)
    print("Saved full-result diagnostics, provenance and summary.")
if __name__=="__main__":run()
