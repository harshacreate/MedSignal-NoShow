from __future__ import annotations
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score)

def fit_models(X,y,seed=42):
    baseline=DummyClassifier(strategy="prior")
    stronger=HistGradientBoostingClassifier(max_iter=140,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed)
    baseline.fit(X,y);stronger.fit(X,y)
    return {"baseline_prior":baseline,"hist_gradient_boosting":stronger}

def evaluate(model,X,y):
    y=np.asarray(y);pred=model.predict(X);labels=np.unique(y)
    result={"n":int(len(y)),"accuracy":round(float(accuracy_score(y,pred)),4),"balanced_accuracy":round(float(balanced_accuracy_score(y,pred)),4),"macro_f1":round(float(f1_score(y,pred,average="macro",zero_division=0)),4),"confusion_matrix":confusion_matrix(y,pred,labels=labels).tolist(),"labels":[int(v) if np.issubdtype(type(v),np.integer) else str(v) for v in labels],"error_count":int(np.sum(pred!=y))}
    if len(labels)==2:
        proba=model.predict_proba(X)[:,1]
        result.update({"roc_auc":round(float(roc_auc_score(y,proba)),4),"average_precision":round(float(average_precision_score(y,proba)),4),"precision_at_0_5":round(float(precision_score(y,pred,zero_division=0)),4),"recall_at_0_5":round(float(recall_score(y,pred,zero_division=0)),4),"f1_at_0_5":round(float(f1_score(y,pred,zero_division=0)),4)})
    return result

def subgroup_metrics(model,X,y,groups):
    pred=model.predict(X);y=np.asarray(y);groups=np.asarray(groups);result={};rates=[];recalls=[]
    for value in np.unique(groups):
        mask=groups==value;rate=float(np.mean(pred[mask]==1)) if len(np.unique(y))==2 else None;rec=float(recall_score(y[mask],pred[mask],zero_division=0)) if len(np.unique(y))==2 else None
        result[str(value)]={"n":int(mask.sum()),"label_rate":round(float(np.mean(y[mask]==1)),4),"prediction_rate":None if rate is None else round(rate,4),"recall":None if rec is None else round(rec,4)}
        if rate is not None:rates.append(rate)
        if rec is not None:recalls.append(rec)
    return {"groups":result,"prediction_rate_gap":round(max(rates)-min(rates),4) if len(rates)>1 else None,"recall_gap":round(max(recalls)-min(recalls),4) if len(recalls)>1 else None}
