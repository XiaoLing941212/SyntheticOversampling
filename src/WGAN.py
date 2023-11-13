"""
This file contains necessary WGAN and related functions

Reference: https://github.com/LyudaK/msc_thesis_imblearn/blob/main/ImbLearn.ipynb
Modified by: Xiao Ling, xling4@ncsu.edu
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import time

class Generator(tf.keras.Model):
    def __init__(self, output_dim, input_dim, hidden_dim=128):
        super().__init__()
        init = tf.keras.initializers.GlorotUniform
        self.input_layer = tf.keras.layers.Dense(units=input_dim, kernel_initializer=init)
        self.hidden_layer = tf.keras.layers.Dense(units=hidden_dim, activation="relu", kernel_initializer=init)
        self.output_layer = tf.keras.layers.Dense(units=output_dim, activation="sigmoid", kernel_initializer=init)
    
    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        return self.output_layer(x)


class Critic(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        init = tf.keras.initializers.GlorotUniform
        self.input_layer = tf.keras.layers.Dense(units=input_dim, kernel_initializer=init)
        self.hidden_layer = tf.keras.layers.Dense(units=hidden_dim, kernel_initializer=init)
        self.logits = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer=init)

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_layer(x)
        return self.logits(x)
    

@tf.function
def train_step(real_data, gen, critic, noise_dim, generator_optimizer, critic_optimizer):
    batch_size = real_data.shape[0]
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
        fake_data = gen(noise, training=True)
        real_output = critic(real_data, training=True)
        fake_output = critic(fake_data, training=True)

        critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        critic_loss_real = tf.reduce_mean(real_output)
        critic_loss_fake = tf.reduce_mean(fake_output)
        gen_loss = -tf.reduce_mean(fake_output)
    
    wasserstein = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_critic = critic_tape.gradient(critic_loss, critic.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    tf.group(*(var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in critic.trainable_variables))

    return wasserstein, gen_loss, critic_loss_real, critic_loss_fake


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


def WGANOversampling(X_train, y_train):
    tar = y_train.name

    start_time = time.time()

    X_sample = gen_data(X_train=X_train, y_train=y_train,
                        target=tar, classes=list(y_train.unique()))
    
    X_train[tar] = y_train
    X_sample = X_sample.append(X_train)
    y_sample = X_sample[tar]
    X_sample = X_sample.drop(tar, 1)

    return round(time.time() - start_time, 2), X_sample, y_sample