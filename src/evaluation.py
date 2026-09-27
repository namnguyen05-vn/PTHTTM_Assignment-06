import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, average_precision_score, log_loss

def classification_metrics(labels,logits):
    probability=softmax(logits,axis=1)
    prediction=probability.argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(
        labels,prediction,labels=[0,1,2],zero_division=0)
    result=dict(accuracy=float(accuracy_score(labels,prediction)),
        macro_f1=float(f1_score(labels,prediction,labels=[0,1,2],average="macro",zero_division=0)),
        log_loss=float(log_loss(labels,probability,labels=[0,1,2])),
        per_class=[dict(label=i,precision=float(precision[i]),recall=float(recall[i]),
                       f1=float(f1[i]),support=int(support[i])) for i in range(3)])
    for i in [1,2]:
        result["ap_"+str(i)]=float(average_precision_score(labels==i,probability[:,i])) if np.any(labels==i) else None
    result["confusion_matrix"]=confusion_matrix(labels,prediction,labels=[0,1,2]).tolist()
    return result

def regression_metrics(labels,prediction,base_close=None,target_close=None):
    labels=np.asarray(labels,dtype="float64")
    prediction=np.asarray(prediction,dtype="float64").reshape(-1)
    errors=prediction-labels
    result=dict(mae=float(np.abs(errors).mean()),rmse=float(np.sqrt(np.mean(errors**2))),
        directional_accuracy=float(np.mean(np.sign(prediction)==np.sign(labels))),
        mean_true_return=float(labels.mean()),mean_predicted_return=float(prediction.mean()))
    if base_close is not None:
        clipped=np.clip(prediction,-20,20)
        price=np.asarray(base_close)*np.exp(clipped)
        result.update(price_mae=float(np.abs(price-target_close).mean()),
            price_rmse=float(np.sqrt(np.mean((price-target_close)**2))),
            price_reconstruction_clipped=int(np.sum(clipped!=prediction)))
    return result

def evaluate(data,ids,output):
    if data["metadata"]["dataset"]=="retailrocket":
        return classification_metrics(data["labels"][ids],output)
    prediction=output[:,0]*data["metadata"]["target_scale"]
    return regression_metrics(data["labels"][ids],prediction,
        data["base_close"][ids],data["target_close"][ids])
