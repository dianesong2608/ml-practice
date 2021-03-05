import numpy as np
import pandas as pd
import warnings

def accuracy_score(y_true, y_predict):
    if len(y_true) != len(y_predict):
        raise ValueError("The dimension of predict data and actual data should be the same.")

    return np.sum(y_true == y_predict) / len(y_true)

def confusion_matrix(y_true, y_predict):
    if len(y_true) != len(y_predict):
        raise ValueError("The dimension of predict data and actual data should be the same.")

    return pd.crosstab(y_true, y_predict)

def precision(y_true, y_predict):
    cm = confusion_matrix(y_true, y_predict)
    cols_num = np.sum(cm, axis=0)
    return np.diagonal(cm) / cols_num

def recall(y_true, y_predict):
    cm = confusion_matrix(y_true, y_predict)
    rows_num = np.sum(cm, axis=1)
    return np.diagonal(cm) / rows_num

def f1_score(y_true, y_predict):
    pre = precision(y_true, y_predict)
    rec = recall(y_true, y_predict)
    try:
        return 2 * pre * rec / (pre + rec)
    except:
        return 0.0