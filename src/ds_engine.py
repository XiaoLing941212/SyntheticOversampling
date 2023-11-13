from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

import os
import time
import pandas as pd
import numpy as np

def DSOversampling(X_train, y_train):
    col = X_train.columns
    tar = y_train.name
    X_train[tar] = y_train
    write_df = X_train[X_train[tar] == 1]
    write_df = write_df.iloc[:, :-1]
    write_df.to_csv(f"{os.getcwd()}/extra/ds_df.csv", index=False)
    X_train = X_train.iloc[:, :-1]

    threshold = 20
    num_tuples_to_generate = int(y_train.value_counts()[0] - y_train.value_counts()[1])

    start_time = time.time()

    description_path = f"{os.getcwd()}/extra/ds.json"
    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_independent_attribute_mode(
        dataset_file=f"{os.getcwd()}/extra/ds_df.csv"
    )
    describer.save_dataset_description_to_file(description_path)

    generator = DataGenerator()
    generator.generate_dataset_in_independent_mode(num_tuples_to_generate, description_path)
    generator.save_synthetic_data(f"{os.getcwd()}/extra/ds_syn_df.csv")

    rt = time.time() - start_time

    X_train_new = pd.read_csv(f"{os.getcwd()}/extra/ds_syn_df.csv").to_numpy()
    y_train_new = np.ones(num_tuples_to_generate)
    X_train_new = pd.DataFrame(np.vstack((X_train.to_numpy(), X_train_new)), columns=col)
    y_train_new = pd.Series(np.hstack((y_train.to_numpy(), y_train_new)), name=tar)

    return rt, X_train_new, y_train_new