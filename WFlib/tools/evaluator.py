import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def measurement(y_true, y_pred, eval_metrics):
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred)*100, 2)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro")*100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro")*100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro")*100, 2)
        elif eval_metric == "P@min":
            results[eval_metric] = np.min(precision_score(y_true, y_pred, average=None))
        elif eval_metric == "r-Precision":
            results[eval_metric] = cal_r_precision(y_true, y_pred)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results

def cal_r_precision(y_true, y_pred, base_r=20):
    open_class = y_true.max()
    web2tp = {}
    web2fp = {}
    web2wp = {}
    for web in range(open_class+1):
        web2tp[web] = 0
        web2fp[web] = 0
        web2wp[web] = 0

    for index in range(len(y_true)):
        cur_true = y_true[index]
        cur_pred = y_pred[index]
        if cur_true == cur_pred:
            web2tp[cur_pred] += 1
        else:
            if cur_true == open_class:
                web2fp[cur_pred] += 1
            else:
                web2wp[cur_pred] += 1

    res = 0
    for web in range(open_class):
        if (web2tp[web] + base_r*web2fp[web] + web2wp[web]) == 0:
            continue
        res += web2tp[web] / (web2tp[web] + base_r*web2fp[web] + web2wp[web])
    res /= open_class
    return res