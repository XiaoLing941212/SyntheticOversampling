from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score

import math
import numpy as np
import os
from scipy.io import arff
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

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


def parse_results(res):
    res = res.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    res = res.replace('array', '').replace(' ', '')
    res_to_list = res.split(",")
    res_to_list = [float(i) for i in res_to_list]
    res_to_list[0] = int(np.round_(res_to_list[0]))
    res_to_list[1] = int(np.round_(res_to_list[1]))
    res_to_list[2] = int(np.round_(res_to_list[2]))

    return res_to_list


def read_data(project):
    if project == "Moodle_Vuln":
        data_path = f"{os.getcwd()}/data/Vulnerable_Files/moodle-2_0_0-metrics.arff"
        data = arff.loadarff(data_path)
        df = pd.DataFrame(data[0])
        df['IsVulnerable'] = df['IsVulnerable'].astype('str')
        d = {'b\'yes\'': 1, 'b\'no\'': 0}
        df['IsVulnerable'] = df['IsVulnerable'].astype(str).map(d).fillna(df['IsVulnerable'])
        df = df.drop_duplicates()
        df.reset_index(inplace=True, drop=True)

    return df


def create_models():
    clf_SVM = SVC()
    clf_KNN = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf_LR = LogisticRegression(random_state=42, solver="saga", max_iter=20000, n_jobs=-1)
    clf_DT = DecisionTreeClassifier()
    clf_RF = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf_LightGBM = LGBMClassifier(objective="binary", random_state=42, n_jobs=-1)
    clf_Adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf_GBDT = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)

    return clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT