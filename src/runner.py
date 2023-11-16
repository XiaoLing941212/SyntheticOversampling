import pandas as pd
import csv
import os
import sys
import time
import numpy as np
import random
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import evaluate_result, read_data, create_models
from smote_oversampling import RandomOversampling
from smote_oversampling import ADASYNOversampling
from smote_oversampling import BorderlineSMOTEOversampling
from smote_oversampling import SMOTEOversampling
from smote_oversampling import SVMSMOTEOversampling
from smote_oversampling import SMOTUNEDOversampling
from dazzle import DAZZLEOversampling
from WGAN import WGANOversampling
from random_projection import RandomProjectionOversampling
from howso_engine import howsoOversampling
from ds_engine import DSOversampling
from sdv_engine import SDVOversampling

def main(project, repeats=10, rp_threshold=12):
    rs_list = random.sample(range(50, 500), repeats)

    for repeat in range(repeats):
        print(f"----- in repeat {repeat+1} -----")
        rs = rs_list[repeat]

        write_path = f"{project}_res_r{repeat+1}_rn{rs}.csv"
        write_path = f"{os.getcwd()}/result/{project}/{write_path}"
        with open(write_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["oversampling_scheme", "runtime", "learner", "acc", "prec", "recall", "fpr", "f1", "auc", "g_score", "d2h"])

        if project != "Ambari_Vuln":
            df = read_data(project)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print("y value counts: \n", str(y.value_counts()))
            print("y class ratio: 1:", str(round(y.value_counts()[0]/y.value_counts()[1])))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)
            print("--- y train classes count: \n" + str(y_train.value_counts()))
            print("--- y train ratio: 1:" + str(round(y_train.value_counts()[0] / y_train.value_counts()[1])))
            print(" ")
            print("--- y test classes count: \n" + str(y_test.value_counts()))
            print("--- y test ratio: 1:" + str(round(y_test.value_counts()[0] / y_test.value_counts()[1])))
        else:
            train_df, test_df = read_data(project)
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]
            print("--- y train classes count: \n" + str(y_train.value_counts()))
            print("--- y train ratio: 1:" + str(round(y_train.value_counts()[0] / y_train.value_counts()[1])))
            print(" ")
            print("--- y test classes count: \n" + str(y_test.value_counts()))
            print("--- y test ratio: 1:" + str(round(y_test.value_counts()[0] / y_test.value_counts()[1])))

        ### normal run ###
        print("----- normal -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_copy), 
                                     columns=X_train_copy.columns,
                                     index=X_train_copy.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), 
                                    columns=X_test.columns, 
                                    index=X_test.index)
        
        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_copy)
        clf_KNN.fit(X_train_scale, y_train_copy)
        clf_LR.fit(X_train_scale, y_train_copy)
        clf_DT.fit(X_train_scale, y_train_copy)
        clf_RF.fit(X_train_scale, y_train_copy)
        clf_LightGBM.fit(X_train_scale, y_train_copy)
        clf_Adaboost.fit(X_train_scale, y_train_copy)
        clf_GBDT.fit(X_train_scale, y_train_copy)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f) 

            csv_writer.writerow(["No", 0, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["No", 0, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["No", 0, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["No", 0, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["No", 0, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["No", 0, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["No", 0, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["No", 0, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))
        
        ### random run ###
        print("----- random -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = RandomOversampling(X_train=X_train_copy,
                                                          y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))
        
        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["Random", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["Random", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["Random", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["Random", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["Random", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["Random", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["Random", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["Random", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### SMOTE run ###
        print("----- SMOTE -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = SMOTEOversampling(X_train=X_train_copy, 
                                                         y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["SMOTE", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["SMOTE", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["SMOTE", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["SMOTE", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["SMOTE", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["SMOTE", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["SMOTE", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["SMOTE", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))
        
        ### ADASYN run ###
        print("----- ADASYN ------")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = ADASYNOversampling(X_train=X_train_copy, 
                                                          y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["ADASYN", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["ADASYN", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["ADASYN", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["ADASYN", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["ADASYN", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["ADASYN", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["ADASYN", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["ADASYN", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))
        
        ### BorderlineSMOTE run ###
        print("----- borderlineSMOTE -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = BorderlineSMOTEOversampling(X_train=X_train_copy, 
                                                                   y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["BorderlineSMOTE", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["BorderlineSMOTE", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### SVMSMOTE run ###
        print("----- SVMSMOTE -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = SVMSMOTEOversampling(X_train=X_train_copy, 
                                                            y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["SVMSMOTE", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["SVMSMOTE", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### SMOTUNED run ###
        print("----- SMOTUNED -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()
        X_test_copy, y_test_copy = X_test.copy(), y_test.copy()

        rt_SVM, X_train_new_SVM, y_train_new_SVM = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                        X_test=X_test_copy, 
                                                                        y_train=y_train_copy, 
                                                                        y_test=y_test_copy, 
                                                                        model="SVM")
        
        scaler = StandardScaler()
        X_train_scale_SVM = pd.DataFrame(scaler.fit_transform(X_train_new_SVM), columns=X_train_new_SVM.columns)
        X_test_scale_SVM = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of SVM: 1:" + str(round(y_train_new_SVM.value_counts()[0] / y_train_new_SVM.value_counts()[1])))

        rt_KNN, X_train_new_KNN, y_train_new_KNN = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                        X_test=X_test_copy, 
                                                                        y_train=y_train_copy, 
                                                                        y_test=y_test_copy, 
                                                                        model="KNN")

        scaler = StandardScaler()
        X_train_scale_KNN = pd.DataFrame(scaler.fit_transform(X_train_new_KNN), columns=X_train_new_KNN.columns)
        X_test_scale_KNN = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of KNN: 1:" + str(round(y_train_new_KNN.value_counts()[0] / y_train_new_KNN.value_counts()[1])))

        rt_LR, X_train_new_LR, y_train_new_LR = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                     X_test=X_test_copy, 
                                                                     y_train=y_train_copy, 
                                                                     y_test=y_test_copy, 
                                                                     model="LR")

        scaler = StandardScaler()
        X_train_scale_LR = pd.DataFrame(scaler.fit_transform(X_train_new_LR), columns=X_train_new_LR.columns)
        X_test_scale_LR = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of LR: 1:" + str(round(y_train_new_LR.value_counts()[0] / y_train_new_LR.value_counts()[1])))

        rt_DT, X_train_new_DT, y_train_new_DT = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                     X_test=X_test_copy, 
                                                                     y_train=y_train_copy, 
                                                                     y_test=y_test_copy, 
                                                                     model="DT")

        scaler = StandardScaler()
        X_train_scale_DT = pd.DataFrame(scaler.fit_transform(X_train_new_DT), columns=X_train_new_DT.columns)
        X_test_scale_DT = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of DT: 1:" + str(round(y_train_new_DT.value_counts()[0] / y_train_new_DT.value_counts()[1])))

        rt_RF, X_train_new_RF, y_train_new_RF = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                     X_test=X_test_copy, 
                                                                     y_train=y_train_copy, 
                                                                     y_test=y_test_copy, 
                                                                     model="RF")

        scaler = StandardScaler()
        X_train_scale_RF = pd.DataFrame(scaler.fit_transform(X_train_new_RF), columns=X_train_new_RF.columns)
        X_test_scale_RF = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of RF: 1:" + str(round(y_train_new_RF.value_counts()[0] / y_train_new_RF.value_counts()[1])))

        rt_LightGBM, X_train_new_LightGBM, y_train_new_LightGBM = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                                       X_test=X_test_copy, 
                                                                                       y_train=y_train_copy, 
                                                                                       y_test=y_test_copy, 
                                                                                       model="LightGBM")

        scaler = StandardScaler()
        X_train_scale_LightGBM = pd.DataFrame(scaler.fit_transform(X_train_new_LightGBM), columns=X_train_new_LightGBM.columns)
        X_test_scale_LightGBM = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of LightGBM: 1:" + str(round(y_train_new_LightGBM.value_counts()[0] / y_train_new_LightGBM.value_counts()[1])))

        rt_Adaboost, X_train_new_Adaboost, y_train_new_Adaboost = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                                       X_test=X_test_copy, 
                                                                                       y_train=y_train_copy, 
                                                                                       y_test=y_test_copy, 
                                                                                       model="Adaboost")

        scaler = StandardScaler()
        X_train_scale_Adaboost = pd.DataFrame(scaler.fit_transform(X_train_new_Adaboost), columns=X_train_new_Adaboost.columns)
        X_test_scale_Adaboost = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of Adaboost: 1:" + str(round(y_train_new_Adaboost.value_counts()[0] / y_train_new_Adaboost.value_counts()[1])))

        rt_GBDT, X_train_new_GBDT, y_train_new_GBDT = SMOTUNEDOversampling(X_train=X_train_copy, 
                                                                           X_test=X_test_copy, 
                                                                           y_train=y_train_copy, 
                                                                           y_test=y_test_copy, 
                                                                           model="GBDT")

        scaler = StandardScaler()
        X_train_scale_GBDT = pd.DataFrame(scaler.fit_transform(X_train_new_GBDT), columns=X_train_new_GBDT.columns)
        X_test_scale_GBDT = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        print("y train ratio of GBDT: 1:" + str(round(y_train_new_GBDT.value_counts()[0] / y_train_new_GBDT.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale_SVM, y_train_new_SVM)
        clf_KNN.fit(X_train_scale_KNN, y_train_new_KNN)
        clf_LR.fit(X_train_scale_LR, y_train_new_LR)
        clf_DT.fit(X_train_scale_DT, y_train_new_DT)
        clf_RF.fit(X_train_scale_RF, y_train_new_RF)
        clf_LightGBM.fit(X_train_scale_LightGBM, y_train_new_LightGBM)
        clf_Adaboost.fit(X_train_scale_Adaboost, y_train_new_Adaboost)
        clf_GBDT.fit(X_train_scale_GBDT, y_train_new_GBDT)

        y_pred_SVM = clf_SVM.predict(X_test_scale_SVM)
        y_pred_KNN = clf_KNN.predict(X_test_scale_KNN)
        y_pred_LR = clf_LR.predict(X_test_scale_LR)
        y_pred_DT = clf_DT.predict(X_test_scale_DT)
        y_pred_RF = clf_RF.predict(X_test_scale_RF)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale_LightGBM)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale_Adaboost)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale_GBDT)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["SMOTUNED", rt_SVM, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["SMOTUNED", rt_KNN, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["SMOTUNED", rt_LR, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["SMOTUNED", rt_DT, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["SMOTUNED", rt_RF, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["SMOTUNED", rt_LightGBM, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["SMOTUNED", rt_Adaboost, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["SMOTUNED", rt_GBDT, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### DAZZLE run ###
        print("----- DAZZLE -----")
        cols = X_train.columns
        tar = y_train.name

        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()
        X_test_copy, y_test_copy = X_test.copy(), y_test.copy()

        rt, X_train_new, y_train_new, X_test_scale = DAZZLEOversampling(X_train=X_train_copy, 
                                                                        y_train=y_train_copy, 
                                                                        X_test=X_test_copy,
                                                                        y_test=y_test_copy)

        X_train_new = pd.DataFrame(X_train_new, columns=cols)
        y_train_new = pd.Series(y_train_new, name=tar)
        X_test_scale = pd.DataFrame(X_test_scale, columns=cols)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_new, y_train_new)
        clf_KNN.fit(X_train_new, y_train_new)
        clf_LR.fit(X_train_new, y_train_new)
        clf_DT.fit(X_train_new, y_train_new)
        clf_RF.fit(X_train_new, y_train_new)
        clf_LightGBM.fit(X_train_new, y_train_new)
        clf_Adaboost.fit(X_train_new, y_train_new)
        clf_GBDT.fit(X_train_new, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["DAZZLE", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["DAZZLE", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["DAZZLE", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["DAZZLE", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["DAZZLE", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["DAZZLE", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["DAZZLE", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["DAZZLE", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### WGAN run ###
        print("----- WGAN -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        scaler = StandardScaler()
        X_train_GAN = pd.DataFrame(scaler.fit_transform(X_train_copy), columns=X_train_copy.columns, index=X_train_copy.index)
        rt, X_train_new, y_train_new = WGANOversampling(X_train=X_train_GAN, 
                                                        y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["WGAN", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["WGAN", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["WGAN", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["WGAN", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["WGAN", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["WGAN", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["WGAN", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["WGAN", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### random projection run ###
        print("----- Random Projection -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = RandomProjectionOversampling(X_train=X_train_copy, 
                                                                    y_train=y_train_copy,
                                                                    threshold=rp_threshold)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["RP", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["RP", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["RP", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["RP", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["RP", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["RP", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["RP", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["RP", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### Howso run ###
        print("----- Howso -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = howsoOversampling(X_train=X_train_copy,
                                                         y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["Howso", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["Howso", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["Howso", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["Howso", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["Howso", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["Howso", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["Howso", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["Howso", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### DS run ###
        print("----- DS -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = DSOversampling(X_train=X_train_copy,
                                                      y_train=y_train_copy)
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["DS", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["DS", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["DS", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["DS", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["DS", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["DS", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["DS", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["DS", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### SDV GC run ###
        print("----- SDV GC -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = SDVOversampling(X_train=X_train_copy,
                                                       y_train=y_train_copy,
                                                       mode="GC")
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["SDV_GC", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["SDV_GC", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["SDV_GC", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["SDV_GC", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["SDV_GC", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["SDV_GC", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["SDV_GC", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["SDV_GC", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        ### SDV GAN run ###
        print("----- SDV GAN -----")
        X_train_copy, y_train_copy = X_train.copy(), y_train.copy()

        rt, X_train_new, y_train_new = SDVOversampling(X_train=X_train_copy,
                                                       y_train=y_train_copy,
                                                       mode="GAN")
        
        scaler = StandardScaler()
        X_train_scale = pd.DataFrame(scaler.fit_transform(X_train_new), columns=X_train_new.columns, index=X_train_new.index)
        X_test_scale = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        print("y train ratio: 1:" + str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))

        clf_SVM, clf_KNN, clf_LR, clf_DT, clf_RF, clf_LightGBM, clf_Adaboost, clf_GBDT = create_models()
        clf_SVM.fit(X_train_scale, y_train_new)
        clf_KNN.fit(X_train_scale, y_train_new)
        clf_LR.fit(X_train_scale, y_train_new)
        clf_DT.fit(X_train_scale, y_train_new)
        clf_RF.fit(X_train_scale, y_train_new)
        clf_LightGBM.fit(X_train_scale, y_train_new)
        clf_Adaboost.fit(X_train_scale, y_train_new)
        clf_GBDT.fit(X_train_scale, y_train_new)

        y_pred_SVM = clf_SVM.predict(X_test_scale)
        y_pred_KNN = clf_KNN.predict(X_test_scale)
        y_pred_LR = clf_LR.predict(X_test_scale)
        y_pred_DT = clf_DT.predict(X_test_scale)
        y_pred_RF = clf_RF.predict(X_test_scale)
        y_pred_LightGBM = clf_LightGBM.predict(X_test_scale)
        y_pred_Adaboost = clf_Adaboost.predict(X_test_scale)
        y_pred_GBDT = clf_GBDT.predict(X_test_scale)

        with open(write_path, "a", newline="") as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(["SDV_GAN", rt, "SVM"] + evaluate_result(y_pred_SVM, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "KNN"] + evaluate_result(y_pred_KNN, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "LR"] + evaluate_result(y_pred_LR, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "DT"] + evaluate_result(y_pred_DT, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "RF"] + evaluate_result(y_pred_RF, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "LightGBM"] + evaluate_result(y_pred_LightGBM, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "Adaboost"] + evaluate_result(y_pred_Adaboost, y_test))
            csv_writer.writerow(["SDV_GAN", rt, "GBDT"] + evaluate_result(y_pred_GBDT, y_test))

        print("----- end of experiment ------")


if __name__ == "__main__":
    case_to_run = sys.argv[1]
    repeats = int(sys.argv[2])
    rp_threshold = int(sys.argv[3])

    main(case_to_run, repeats=repeats ,rp_threshold=rp_threshold)