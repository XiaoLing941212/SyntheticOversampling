import csv
import pickle
from timeit import default_timer as timer
import os
import time

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from models import WGANGP

def get_g_score(y_test, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    recall = recall_score(y_test, y_predict)
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)
    return g_score

def get_cat_dims(X, cat_cols) -> list:
    return [(X[col].nunique()) for col in cat_cols]

def DAZZLEOversampling(X_train, y_train, X_test, y_test):
    def objective(params, ):
        global ITERATION

        ITERATION += 1

        start = timer()
        gan = WGANGP(write_to_disk=False,
                    compute_metrics_every=1250, print_every=50000, plot_every=200000,
                    num_cols=num_cols,
                    cat_dims=cat_dims,
                    transformer=prep.named_transformers_['cat']['onehotencoder'],
                    cat_cols=cat_cols,
                    d_updates_per_g=int(params['d_updates_per_g']),
                    gp_weight=int(params['gp_weight'])
                    )

        gan.fit(X_train_trans, y=y_train.values,
                condition=True,
                epochs=int(params['epochs']),
                batch_size=params['batch_size'],
                netG_kwargs={'hidden_layer_sizes': (128, 64),
                            'n_cross_layers': 1,
                            'cat_activation': 'gumbel_softmax',
                            'num_activation': 'none',
                            'condition_num_on_cat': True,
                            'noise_dim': 30,
                            'normal_noise': False,
                            'activation': params['G_activation'],
                            'reduce_cat_dim': False,
                            'use_num_hidden_layer': True,
                            'layer_norm': params['G_layer_norm'], },
                netD_kwargs={'hidden_layer_sizes': (128, 64, 32),
                            'n_cross_layers': 2,
                            'embedding_dims': 'auto',
                            'activation': params['D_activation'],
                            'sigmoid_activation': False,
                            'noisy_num_cols': params['noisy_num_cols'],
                            'layer_norm': params['D_layer_norm'], }
                )

        X_res, y_res = gan.resample(X_train_trans, y=y_train)

        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        clf.fit(X_res, y_res)
        preds_oversampled = clf.predict(X_test_trans)
        g_score = get_g_score(y_test, preds_oversampled)
        loss = 1 - g_score
        run_time = timer() - start
        return {'loss': loss, 'params': params, 'iteration': ITERATION, 'run_time': run_time, 'status': STATUS_OK}

    space = {
        'd_updates_per_g': hp.quniform('d_updates_per_g', 2, 10, 1),
        'gp_weight': hp.quniform('gp_weight', 5, 20, 1),
        'epochs': hp.quniform('epochs', 100, 400, 1),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'G_activation': hp.choice('G_activation', ['relu', 'leaky_relu', 'tanh', 'sigmoid']),
        'D_activation': hp.choice('D_activation', ['relu', 'leaky_relu', 'tanh', 'sigmoid']),
        'noisy_num_cols': hp.choice('noisy_num_cols', ['True', 'False']),
        'G_layer_norm': hp.choice('G_layer_norm', ['True', 'False']),
        'D_layer_norm': hp.choice('D_layer_norm', ['True', 'False'])
    }

    start_time = time.time()
    num_cols = X_train.columns
    cat_cols = []
    cat_dims = get_cat_dims(X_train, cat_cols)

    num_prep = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler())
    cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'),
                            OneHotEncoder(handle_unknown='ignore', sparse=False))
    prep = ColumnTransformer([('num', num_prep, num_cols), ('cat', cat_prep, cat_cols)], remainder='drop')

    # X_train_train, X_valid, y_train_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    # X_train_trans = prep.fit_transform(X_train_train)
    # X_test_trans = prep.transform(X_valid)
    X_train_trans = prep.fit_transform(X_train)
    X_test_trans = prep.transform(X_test)

    # X_trans = np.vstack((X_train_trans, X_test_trans))
    # y_trans = pd.concat([y_train_train, y_valid], axis=0)

    global ITERATION
    ITERATION = 0

    bayes_trials = Trials()
    MAX_EVALS = 5
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials)

    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])

    gan = WGANGP(write_to_disk=False,
                 compute_metrics_every=1250, print_every=5000, plot_every=10000,
                 num_cols=num_cols,
                 cat_dims=cat_dims,
                 transformer=prep.named_transformers_['cat']['onehotencoder'],
                 cat_cols=cat_cols,
                 d_updates_per_g=int(bayes_trials_results[0]["params"]['d_updates_per_g']),
                 gp_weight=int(bayes_trials_results[0]["params"]['gp_weight']))

    gan.fit(X_train_trans, y=y_train.values,
            condition=True,
            epochs=int(bayes_trials_results[0]["params"]['epochs']),
            batch_size=bayes_trials_results[0]["params"]['batch_size'],
            netG_kwargs={'hidden_layer_sizes': (128, 64),
                            'n_cross_layers': 1,
                            'cat_activation': 'gumbel_softmax',
                            'num_activation': 'none',
                            'condition_num_on_cat': True,
                            'noise_dim': 30,
                            'normal_noise': False,
                            'activation': bayes_trials_results[0]["params"]['G_activation'],
                            'reduce_cat_dim': False,
                            'use_num_hidden_layer': True,
                            'layer_norm': bayes_trials_results[0]["params"]['G_layer_norm'], },
            netD_kwargs={'hidden_layer_sizes': (128, 64, 32),
                            'n_cross_layers': 2,
                            'embedding_dims': 'auto',
                            'activation': bayes_trials_results[0]["params"]['D_activation'],
                            'sigmoid_activation': False,
                            'noisy_num_cols': bayes_trials_results[0]["params"]['noisy_num_cols'],
                            'layer_norm': bayes_trials_results[0]["params"]['D_layer_norm'], }
            )

    X_res, y_res = gan.resample(X_train_trans, y=y_train)

    return time.time() - start_time, X_res, y_res, X_test_trans

# num_cols = ['nonecholoc', 'loc', 'nmethods', 'ccomdeep', 'ccom', 'nest', 'hvol', 'nIncomingCalls',
#             'nIncomingCallsUniq',
#             'nOutgoingInternCalls', 'nOutgoingExternFlsCalled', 'nOutgoingExternFlsCalledUniq',
#             'nOutgoingExternCallsUniq']

# data_path = f"{os.path.dirname(os.getcwd())}\\SyntheticData\\data\\Vulnerable_Files\\moodle-2_0_0-metrics.arff"
# data = arff.loadarff(data_path)
# df = pd.DataFrame(data[0])
# df['IsVulnerable'] = df['IsVulnerable'].astype('str')
# d = {'b\'yes\'': 1, 'b\'no\'': 0}
# df['IsVulnerable'] = df['IsVulnerable'].astype(str).map(d).fillna(df['IsVulnerable'])
# df = df.drop_duplicates()
# df.reset_index(inplace=True, drop=True)

# data_path = f"{os.getcwd()}\\data\\JavaScript_Vulnerability\\"
# datafiles = [f for f in os.listdir(data_path) if f.endswith("csv")]
# df = pd.read_csv(f"{data_path}\\{datafiles[0]}")
# drop_columns = ["name", "longname", "path", "full_repo_path", "line", "column", "endline", "endcolumn"]
# df = df.drop(drop_columns, axis=1)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_res, y_res = DAZZLEOversampling(X_train, y_train)

# clf = RandomForestClassifier(random_state=42, n_jobs=-1)
# clf.fit(X_res, y_res)
# preds_oversampled = clf.predict(X_test)
# g_score = get_g_score(y_test, preds_oversampled)
# print(g_score)