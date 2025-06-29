import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from sklearn.metrics import r2_score


def get_confusionmatrix_fnd(preds, labels):
    # label_predicted = np.argmax(preds, axis=1)
    label_predicted = preds
    print(accuracy_score(labels, label_predicted))
    print(classification_report(labels, label_predicted, labels=[0.0, 1.0], target_names=['real', 'fake'], digits=4))
    print(confusion_matrix(labels, label_predicted, labels=[0, 1]))

# 分类评估指标
def metrics(y_label, y_predict):
    scores = {}
    if y_predict is None or y_label is None:
        print(y_predict, y_label)
    # scores['auc'] = round(roc_auc_score(y_label, y_predict, average='macro'), 4)
    y_predict = np.around(np.array(y_predict)).astype(int)
    scores['f1'] = round(f1_score(y_label, y_predict, average='macro'), 4)
    scores['recall'] = round(recall_score(y_label, y_predict, average='macro'), 4)
    scores['precision'] = round(precision_score(y_label, y_predict, average='macro'), 4)
    scores['acc'] = round(accuracy_score(y_label, y_predict), 4)

    return scores


# 回归评估指标
def reg_metrics(y_true, y_pred):
    reg_scores = {}

    # 确保 y_true 和 y_pred 都是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    reg_scores['mae'] = round(mean_absolute_error(y_true, y_pred), 4)
    reg_scores['mse'] = round(mean_squared_error(y_true, y_pred), 4)
    # reg_scores['r2'] = round(calculate_r2(y_true, y_pred), 4)
    reg_scores['rmse'] = round(calculate_rmse(y_true, y_pred), 4)

    return reg_scores

def calculate_r2(y_true, y_pred):
    """
    计算 R²（决定系数）。

    参数:
    y_true (array-like): 真实值数组。
    y_pred (array-like): 预测值数组。

    返回:
    r2 (float): 决定系数。
    """
    r2 = r2_score(y_true, y_pred)
    return r2

def calculate_rmse(y_true, y_pred):
    """
    计算 RMSE（均方根误差）。

    参数:
    y_true (array-like): 真实值数组。
    y_pred (array-like): 预测值数组。

    返回:
    rmse (float): 均方根误差。
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Huber Loss
def huber_loss(pred, target, delta=1.0):
    residual = torch.abs(pred - target)
    condition = residual < delta
    loss = torch.where(condition, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
    return loss.mean()
