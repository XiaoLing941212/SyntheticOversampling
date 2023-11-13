from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

import time
import pandas as pd
import numpy as np
import os

def SDVOversampling(X_train, y_train, mode):
    col = X_train.columns
    tar = y_train.name
    num_tuples_to_generate = int(y_train.value_counts()[0] - y_train.value_counts()[1])

    X_train[tar] = y_train
    pos_df = X_train[X_train[tar] == 1]
    pos_df = pos_df.iloc[:, :-1]
    X_train = X_train.iloc[:, :-1]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=X_train)
    # metadata.save_to_json(f"{os.getcwd()}/extra/sdv.json")

    start_time = time.time()
    if mode == "GC":
        syn = GaussianCopulaSynthesizer(metadata)
        syn.fit(pos_df)
        X_train_new = syn.sample(num_rows=num_tuples_to_generate)
        print(X_train_new)
        rt = time.time() - start_time
    elif mode == "GAN":
        syn = CTGANSynthesizer(metadata)
        syn.fit(data=pos_df)
        X_train_new = syn.sample(num_rows=num_tuples_to_generate).to_numpy()
        rt = time.time() - start_time
    
    X_train_new = pd.DataFrame(np.vstack((X_train.to_numpy(), X_train_new)), columns=col)
    y_train_new = np.ones(num_tuples_to_generate)
    y_train_new = pd.Series(np.hstack((y_train.to_numpy(), y_train_new)), name=tar)

    return rt, X_train_new, y_train_new