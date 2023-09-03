"""
This file contains necessary TABGAN and related functions

Reference: https://github.com/LyudaK/msc_thesis_imblearn/blob/main/ImbLearn.ipynb
Modified by: Xiao Ling, xling4@ncsu.edu
"""

import pandas as pd
from tabgan.sampler import OriginalGenerator, GANGenerator

def run_tabgan(X_train, y_train, X_test, y_test, target, classes):
    count_classes = dict(y_train[target].value_counts())
    max_class = max(count_classes.values)

    new_train = pd.DataFrame()
    new_target = pd.Series()

    tmp = X_train.copy()
    tmp[target] = y_train
    tmp_test = X_test.copy()
    tmp_test[target] = y_test

    for c in set(classes):
        training_data = tmp[tmp[target] == c]
        print("CLASS", c)

        nb_instances_to_generate = 1 + max_class / count_classes[c]

        if nb_instances_to_generate != 1:
            new_tr, new_tar = GANGenerator(
                gen_x_times=nb_instances_to_generate,
                cat_cols=None,
                bot_filter_quantile=0.001,
                top_filter_quantile=0.999,
                is_post_process=False,
                adversarial_model_params={
                    "metrics": "AUC",
                    "max_depth": 2,
                    "max_bin": 100,
                    "learning_rate": 0.02,
                    "random_state": 42,
                    "n_estimators": 500,
                },
                pregeneration_frac=2,
                only_generated_data=True,
                gan_params={
                    "batch_size": 16,
                    "patience": 5,
                    "epochs": 150,
                }
            ).generate_data_pipe(
                pd.DataFrame(training_data.drop(target, 1)),
                pd.DataFrame(training_data[target]),
                tmp_test[tmp_test[target] == c].drop(target, 1),
                deep_copy=True,
                only_adversarial=False,
                use_adversarial=True
            )

            new_train = new_train.append(new_tr)
            new_target = new_target.append(new_tar)
        
    new_target = pd.DataFrame(new_target, columns=[9])
    return new_train, new_target
