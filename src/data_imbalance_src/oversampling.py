from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE

import time

def RandomOversampling(X_train, y_train):
    random_oversampler = RandomOverSampler(random_state=42)
    start_time = time.time()
    X_train_new, y_train_new = random_oversampler.fit_resample(X_train, y_train)

    return round(time.time() - start_time, 2), X_train_new, y_train_new


def ADASYNOversampling(X_train, y_train):
    ADASYN_oversampler = ADASYN(random_state=42)
    start_time = time.time()
    X_train_new, y_train_new = ADASYN_oversampler.fit_resample(X_train, y_train)

    return round(time.time() - start_time, 2), X_train_new, y_train_new


def BorderlineSMOTEOversampling(X_train, y_train):
    BorderlineSMOTE_oversampler = BorderlineSMOTE(random_state=42)
    start_time = time.time()
    X_train_new, y_train_new = BorderlineSMOTE_oversampler.fit_resample(X_train, y_train)

    return round(time.time() - start_time, 2), X_train_new, y_train_new


def SMOTEOversampling(X_train, y_train):
    SMOTE_oversampler = SMOTE(random_state=42)
    start_time = time.time()
    X_train_new, y_train_new = SMOTE_oversampler.fit_resample(X_train, y_train)

    return round(time.time() - start_time, 2), X_train_new, y_train_new


def SVMSMOTEOversampling(X_train, y_train):
    SVMSMOTE_oversampler = SVMSMOTE(random_state=42)
    start_time = time.time()
    X_train_new, y_train_new = SVMSMOTE_oversampler.fit_resample(X_train, y_train)

    return round(time.time() - start_time, 2), X_train_new, y_train_new