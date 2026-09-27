"""Exploration and later result figures; no training in this module."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import ROOT
from src.data import load_data,split_ids
from src.io import read_json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
def save(name):
    plt.tight_layout()
    plt.savefig(FIG/(name+".png"),dpi=180,bbox_inches="tight")
    plt.close()

def explore():
    d=load_data("retailrocket");meta=d["metadata"]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    x=np.arange(3)
    for i,(split,counts) in enumerate(meta["class_counts"].items()):
        axes[0].bar(x+i*.25,counts,width=.25,label=split)
    axes[0].set_xticks(x+.25,meta["labels"])
    axes[0].set_yscale("log");axes[0].set_ylabel("Số mẫu (log)");axes[0].legend()
    axes[1].hist(d["lengths"],bins=np.arange(.5,21.5),color="#315b7c")
    axes[1].set(xlabel="Độ dài lịch sử sử dụng",ylabel="Số mẫu")
    save("retail_distribution")
    days=pd.to_datetime(d["times"],unit="s",utc=True).floor("D")
    table=pd.crosstab(days,np.asarray(d["labels"]))
    table.columns=meta["labels"]
    table.plot(figsize=(10,3.5))
    plt.xlabel("Ngày (UTC)");plt.ylabel("Số nhãn mục tiêu / ngày");save("retail_daily")
    ids=split_ids(d,"train");counts=np.zeros((3,3))
    np.add.at(counts,(d["previous_event"][ids],d["labels"][ids]),1)
    transition=counts/np.maximum(counts.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(5.3,4))
    image=ax.imshow(transition,vmin=0,vmax=1,cmap="Blues")
    ax.set_xticks(range(3),meta["labels"]);ax.set_yticks(range(3),meta["labels"])
    ax.set(xlabel="Hành vi tiếp theo",ylabel="Hành vi gần nhất (train)")
    for row in range(3):
        for col in range(3):ax.text(col,row,f"{transition[row,col]:.3f}",ha="center",va="center",color="white" if transition[row,col]>.5 else "black")
    fig.colorbar(image,ax=ax);save("retail_transition")
    d=load_data("sp500")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for split in ["train","val","test"]:
        y=np.asarray(d["labels"][split_ids(d,split)])
        axes[0].hist(y,bins=80,range=(-.1,.1),density=True,alpha=.4,label=split)
    axes[0].set(xlabel="Log return (khung hiển thị ±0,1)",ylabel="Mật độ trong khung")
    axes[0].legend()
    for ticker in ["AAPL","MSFT","AMZN"]:
        code=d["metadata"]["tickers"].index(ticker)
        ids=np.flatnonzero(d["groups"]==code)
        prices=d["target_close"][ids]
        axes[1].plot(pd.to_datetime(d["times"][ids],unit="s"),prices/prices[0],label=ticker)
    axes[1].set(ylabel="Giá / giá đầu chuỗi hiển thị");axes[1].legend()
    axes[1].tick_params(axis="x",rotation=25);save("stock_exploration")
    counts=np.bincount(d["groups"])
    plt.figure(figsize=(8,3))
    plt.hist(counts,bins=30,color="#315b7c");plt.xlabel("Số cửa sổ hợp lệ / mã");plt.ylabel("Số mã")
    save("stock_windows")

def results():
    for name in ["retailrocket","sp500"]:
        d=load_data(name)
        fig,axes=plt.subplots(1,3 if name=="retailrocket" else 2,figsize=(12 if name=="retailrocket" else 10,3.5))
        for framework in ["pytorch","keras"]:
            folder=ROOT/"results/runs"/(name+"_"+framework+"_seed42")
            history=pd.read_csv(folder/"history.csv")
            axes[0].plot(history.epoch,history.train_loss,label=framework)
            axes[1].plot(history.epoch,history.val_loss,label=framework)
            if name=="retailrocket":
                axes[2].plot(history.epoch,history.val_score,label=framework)
                best=read_json(folder/"metrics.json")["best_epoch"]
                axes[2].scatter([best],[history.loc[history.epoch==best,"val_score"].iloc[0]],marker="*",s=95)
                axes[2].set_ylabel("Validation macro-F1")
        for ax in axes:ax.set_xlabel("Epoch");ax.legend()
        axes[0].set_ylabel("Train CE có trọng số" if name=="retailrocket" else "Train MSE (return chuẩn hóa)")
        axes[1].set_ylabel("Validation CE" if name=="retailrocket" else "Validation RMSE (log return)")
        save(name+"_learning")
        for framework in ["pytorch","keras"]:
            folder=ROOT/"results/runs"/(name+"_"+framework+"_seed42")
            metrics=read_json(folder/"metrics.json")
            if name=="retailrocket":
                cm=np.array(metrics["confusion_matrix"])
                normal=cm/np.maximum(cm.sum(1,keepdims=True),1)
                fig,ax=plt.subplots(figsize=(5.3,4))
                im=ax.imshow(normal,vmin=0,vmax=1,cmap="Blues")
                ax.set_xticks(range(3),d["metadata"]["labels"]);ax.set_yticks(range(3),d["metadata"]["labels"])
                for r in range(3):
                    for c in range(3):ax.text(c,r,str(cm[r,c]),ha="center",va="center",color="white" if normal[r,c]>.5 else "black")
                ax.set(xlabel="Dự đoán",ylabel="Nhãn thật",title=framework);fig.colorbar(im,ax=ax)
                save(name+"_"+framework+"_confusion")
            else:
                frame=pd.read_csv(folder/"predictions.csv")
                code=d["metadata"]["tickers"].index("AAPL")
                subset=frame[frame.group==code].sort_values("timestamp")
                dates=pd.to_datetime(subset.timestamp,unit="s")
                fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
                axes[0].plot(dates,subset.actual_close,label="Thực tế")
                axes[0].plot(dates,subset.predicted_close,label="Dự đoán một phiên",alpha=.8)
                axes[1].plot(dates,subset.actual,label="Return thực")
                axes[1].plot(dates,subset.prediction,label="Return dự đoán",alpha=.8)
                for ax in axes:ax.legend()
                axes[0].set_ylabel("Giá đóng cửa AAPL");axes[1].set_ylabel("Log return")
                save(name+"_"+framework+"_forecast")
                grouped=frame.assign(abs_error=np.abs(frame.prediction-frame.actual)).groupby("group").abs_error.mean()
                pd.DataFrame({"ticker":[d["metadata"]["tickers"][int(i)] for i in grouped.index],
                    "mae":grouped.values}).to_csv(ROOT/"results"/(framework+"_stock_per_ticker.csv"),index=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--results",action="store_true")
    args=parser.parse_args()
    explore()
    if args.results:results()
