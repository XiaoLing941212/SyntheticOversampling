from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score

import math

def evaluate_result(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

    recall = recall_score(y_true=y_true, y_pred=y_pred)
    fpr = fp / (fp + tn)
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)
    d2h = math.sqrt((1 - recall)**2 + (0 - fpr)**2) / math.sqrt(2)

    return [round(accuracy, 3), 
            round(precision, 3), 
            round(recall, 3), 
            round(fpr, 3), 
            round(f1, 3), 
            round(roc_auc, 3), 
            round(g_score, 3), 
            round(d2h, 3)]