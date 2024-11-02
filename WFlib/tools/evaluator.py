import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

def precision_at_k(y_true, y_pred, k):
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    precisions = []

    for i in range(y_true.shape[0]):
        true_positives = np.sum(y_true[i, top_k_preds[i]])
        precisions.append(true_positives / k)

    return np.mean(precisions)

def average_precision_at_k(y_true, y_pred, k):
    res = 0
    for i in range(k):
        res += precision_at_k(y_true, y_pred, i+1)
    res /= k
    return res

def measurement(y_true, y_pred, eval_metrics, num_tabs=1):
    """
    Calculate evaluation metrics for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    eval_metrics (list): List of evaluation metrics to calculate.

    Returns:
    dict: Dictionary of calculated metrics.
    """
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred), 4)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro"), 4)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro"), 4)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro"), 4)
        elif eval_metric == "P@min":
            results[eval_metric] = round(np.min(precision_score(y_true, y_pred, average=None)), 4)
        elif eval_metric == "r-Precision":
            results[eval_metric] = round(cal_r_precision(y_true, y_pred), 4)
        elif eval_metric == "AUC":
            results[eval_metric] = round(roc_auc_score(y_true, y_pred, average='macro'), 4)
        elif eval_metric.startswith("P@"):
            results[eval_metric] = round(precision_at_k(y_true, y_pred, int(eval_metric[-1])), 4)
        elif eval_metric.startswith("AP@"):
            results[eval_metric] = round(average_precision_at_k(y_true, y_pred, int(eval_metric[-1])), 4)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results

def cal_r_precision(y_true, y_pred, base_r=20):
    """
    Calculate r-Precision for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    base_r (int): Base value for r-Precision calculation.

    Returns:
    float: Calculated r-Precision value.
    """
    open_class = y_true.max()  # The class index representing the open world scenario
    web2tp = {}  # True positives per class
    web2fp = {}  # False positives per class
    web2wp = {}  # Wrong predictions per class

    # Initialize dictionaries
    for web in range(open_class + 1):
        web2tp[web] = 0
        web2fp[web] = 0
        web2wp[web] = 0

    # Count true positives, false positives, and wrong predictions
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
    # Calculate r-Precision for each class
    for web in range(open_class):
        denominator = web2tp[web] + base_r * web2fp[web] + web2wp[web]
        if denominator == 0:
            continue
        res += web2tp[web] / denominator
    res /= open_class
    return res

def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad