"""
This file contains data generation functions

Reference: https://github.com/LyudaK/msc_thesis_imblearn/blob/main/ImbLearn.ipynb
Modified by: Xiao Ling, xling4@ncsu.edu
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from data_imbalance_src.Imbalance_Farou2022.WGAN import Generator, Critic, train_step

def generate_synthetic_samples(generator, class_id, headers_name, nb_instance, noise_dim):
    fake_data = generator(tf.random.normal([nb_instance, noise_dim]))

    synthetic_data = pd.DataFrame(data=np.array(fake_data), columns=headers_name)
    synthetic_data["0"] = np.repeat(class_id, len(fake_data))

    return synthetic_data


def fake_data_generation(training_data, nb_instances_to_generate, target):
    BATCH_SIZE = 8
    NOISE_DIM = 10
    learning_rate = 0.001
    epochs = 150

    headers_name = list(training_data.columns.values)
    headers_name = headers_name[:-1]
    class_id = training_data[target].values[0]
    print("CLASS ID", class_id)

    X = training_data.iloc[:, :-1].values.astype("float32")
    n_inp = X.shape[1]
    train_dataset = (tf.data.Dataset.from_tensor_slices(X.reshape(X.shape[0], n_inp)).batch(BATCH_SIZE))

    generator = Generator(n_inp, NOISE_DIM)
    critic = Critic(n_inp)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    # storage
    epoch_wasserstein, epoch_gen_loss, epoch_critic_loss_real, epoch_critic_loss_fake = [], [], [], []

    # start epoch
    for epoch in range(epochs):
        batch_idx, batch_wasserstein, batch_gen, batch_critic_real, batch_critic_fake = 0, 0, 0, 0, 0

        for batch in train_dataset:
            wasserstein, gen_loss, critic_loss_real, critic_loss_fake = train_step(
                real_data=batch,
                gen=generator,
                critic=critic,
                noise_dim=NOISE_DIM,
                generator_optimizer=generator_optimizer,
                critic_optimizer=critic_optimizer
            )

            epoch_wasserstein.append(wasserstein)
            epoch_gen_loss.append(gen_loss)
            epoch_critic_loss_real.append(critic_loss_real)
            epoch_critic_loss_fake.append(critic_loss_fake)
            batch_gen += gen_loss
            batch_critic_real += critic_loss_real
            batch_critic_fake += critic_loss_fake
            batch_wasserstein += wasserstein
            batch_idx += 1
        
        batch_wasserstein = batch_wasserstein / batch_idx
        batch_gen = batch_gen / batch_idx
        batch_critic_real = batch_critic_real / batch_idx
        batch_critic_fake = batch_critic_fake / batch_idx

        if epoch%50 == 0:
            print(f"Epoch {epoch+1}/{epochs} completed. Gen loss: {batch_gen}. Desc loss_real: {batch_critic_real}. Desc loss_fake: {batch_critic_fake}")
    
    data = generate_synthetic_samples(
        generator=generator,
        class_id=class_id,
        headers_name=headers_name,
        nb_instance=nb_instances_to_generate,
        noise_dim=NOISE_DIM
    )

    return data


def gen_data(X_train, y_train, target, classes):
    count_classes = collections.Counter(y_train)
    max_class = max(count_classes.values())
    print("MAX CLASS", max_class)

    new_data = pd.DataFrame()
    tmp = X_train.copy()
    tmp[target] = y_train

    for c in set(classes):
        training_data = tmp[tmp[target] == c]
        nb_instances_to_generate = max_class - count_classes[c]

        if nb_instances_to_generate != 0:
            synthetic_data = fake_data_generation(
                training_data=training_data,
                nb_instances_to_generate=nb_instances_to_generate,
                target=target
            )

            synthetic_data.rename(columns={'0':target}, inplace=True)
            synthetic_data[target] = c
            new_data = new_data.append(synthetic_data)
    
    return new_data


def GANOversampling(X_train, y_train):
    tar = y_train.name

    start_time = time.time()

    X_sample = gen_data(X_train=X_train, y_train=y_train,
                        target=tar, classes=list(y_train.unique()))
    
    X_train[tar] = y_train
    X_sample = X_sample.append(X_train)
    y_sample = X_sample[tar]
    X_sample = X_sample.drop(tar, 1)

    return round(time.time() - start_time, 2), X_sample, y_sample

# data_path = f"{os.getcwd()}\\data\\JavaScript_Vulnerability\\"
# datafiles = [f for f in os.listdir(data_path) if f.endswith("csv")]
# df = pd.read_csv(f"{data_path}\\{datafiles[0]}")
# drop_columns = ["name", "longname", "path", "full_repo_path", "line", "column", "endline", "endcolumn"]
# df = df.drop(drop_columns, axis=1)
# df = df.drop_duplicates()
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)
# X = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# rt, X_train_new, y_train_new = GANOversampling(X_train=X_train, y_train=y_train)
# print(X_train_new)
# print(str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))