import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import time
import pandas as pd
import os

from sklearn.metrics import recall_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, Trials, space_eval

def normalize(sample):
    return (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1E-32) * 2 - 1


def denormalize(generated_sample, original_sample):
    return generated_sample * (np.max(original_sample) - np.min(original_sample) + 1E-32) / 2 + (
            np.max(original_sample) + np.min(original_sample)) / 2


class Generator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, activation_fn, use_layer_norm):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        self.input_layer = tf.keras.layers.Dense(hidden_dim, input_dim=input_dim)
        self.hidden_layer = tf.keras.layers.Dense(hidden_dim*2)
        self.output_layer = tf.keras.layers.Dense(input_dim)
        self.norm_layer1 = tf.keras.layers.LayerNormalization()
        self.norm_layer2 = tf.keras.layers.LayerNormalization()
        self.act_layer = tf.keras.layers.Activation(activation_fn)

    def call(self, inputs):
        x = self.input_layer(inputs)
        if self.use_layer_norm:
            x = self.norm_layer1(x)
        x = self.act_layer(x)
        x = self.hidden_layer(x)
        if self.use_layer_norm:
            x = self.norm_layer2(x)
        x = self.act_layer(x)
        x = self.output_layer(x)

        return x


class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, use_layer_norm):
        super().__init__()
        self.use_layer_norm = use_layer_norm

        self.input_layer = tf.keras.layers.Dense(hidden_dim*2, input_dim=input_dim)
        self.hidden_layer = tf.keras.layers.Dense(hidden_dim)
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        self.norm_layer1 = tf.keras.layers.LayerNormalization()
        self.norm_layer2 = tf.keras.layers.LayerNormalization()
        self.act_layer = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = self.input_layer(inputs)
        if self.use_layer_norm:
            x = self.norm_layer1(x)
        x = self.act_layer(x)
        x = self.hidden_layer(x)
        if self.use_layer_norm:
            x = self.norm_layer2(x)
        x = self.act_layer(x)
        x = self.output_layer(x)

        return x


class WGAN():
    def __init__(self, input_dim, output_dim, hidden_dim,
                 generator_lr, discriminator_lr, 
                 generator_optimizer, discriminator_optimizer,
                 generator_activation_fn, discriminator_activation_fn,
                 generator_layer_normalization, discriminator_layer_normalization,
                 epochs, batch_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.generator_optimizer = generator_optimizer(learning_rate=self.generator_lr)
        self.discriminator_optimizer = discriminator_optimizer(learning_rate=self.discriminator_lr)
        self.generator_activation_fn = generator_activation_fn
        self.discriminator_activation_fn = discriminator_activation_fn
        self.generator_layer_normalization = generator_layer_normalization
        self.discriminator_layer_normalization = discriminator_layer_normalization
        self.epochs = epochs
        self.batch_size = batch_size

        self.generator = Generator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation_fn=generator_activation_fn,
            use_layer_norm=generator_layer_normalization
        )
        self.generator.compile(
            optimizer=self.generator_optimizer,
            loss='binary_crossentropy'
        )
        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=discriminator_layer_normalization
        )
        self.discriminator.compile(
            optimizer=self.discriminator_optimizer,
            loss='binary_corssentropy'
        )

        self.wgan = self.build_wgan()

    def build_wgan(self):
        self.discriminator.trainable = False
        model = tf.keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)

        return model
    
    def gradient_penalty(self, real_data, fake_data):
        epsilon = tf.random.uniform([real_data.shape[0], 1], 0.0, 1.0)
        interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated_data = tf.reshape(interpolated_data,
                                       (interpolated_data.shape[0], self.input_dim))
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_data)
            interpolated_output = self.discriminator(interpolated_data)
        
        gradients = tape.gradient(interpolated_output, interpolated_data)
        gradients_norm = tf.norm(gradients, axis=1)
        gradients_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)

        return gradients_penalty
    
    @ tf.function
    def train_step(self, X_train):
        for epoch in range(self.epochs):      
            noise = tf.random.normal([self.batch_size, self.input_dim])

            choice = np.random.choice(X_train.shape[0], self.batch_size)
            batch = tf.gather(X_train, choice)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_data = self.generator(noise, training=True)
                real_output = self.discriminator(batch, training=True)
                fake_output = self.discriminator(generated_data, training=True)

                generator_loss = -tf.reduce_mean(fake_output)
                discriminator_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gradient_penalty = self.gradient_penalty(batch.numpy(), generated_data.numpy())

                discriminator_loss += 10.0 * gradient_penalty
            
            generator_gradients = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            discrimintor_gradients = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(
                zip(generator_gradients, self.generator.trainable_variables),
            )
            self.discriminator_optimizer.apply_gradients(
                zip(discrimintor_gradients, self.discriminator.trainable_variables)
            )
    

def calculate_g_measure(true_labels, predicted_labels):
    recall = recall_score(y_true=true_labels, y_pred=predicted_labels)
    tn, fp, fn, tp = confusion_matrix(y_true=true_labels, y_pred=predicted_labels).ravel()
    fpr = fp / (fp + tn)
    g_score = 2 * recall * (1 - fpr) / (recall + 1 - fpr)

    return g_score


def convert_activation_fn(identifier):
    activation_fn_mapping = {
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu,
        "tanh": tf.nn.tanh,
        "sigmoid": tf.nn.sigmoid
    }

    return activation_fn_mapping[identifier]


def DAZZLEOversampling(X_train, y_train):
    cols = X_train.columns
    tar = y_train.name

    def objective(hyperparameters):
        hyperparameters["generator_activation_fn"] = convert_activation_fn(
            hyperparameters["generator_activation_fn"]
        )
        hyperparameters["discriminator_activation_fn"] = convert_activation_fn(
            hyperparameters["discriminator_activation_fn"]
        )
        hyperparameters["epochs"] = int(hyperparameters["epochs"])

        wgan = WGAN(X_train.shape[1], 1, 128, **hyperparameters)
        wgan.train_step(X_train=X_train)

        pos_train, neg_train = y_train.value_counts()[1], y_train.value_counts()[0]
        generated_samples = wgan.generator(tf.random.normal([neg_train-pos_train, X_train.shape[1]])).numpy()
        
        combined_samples = np.concatenate((X_train, generated_samples), axis=0)
        true_labels = np.concatenate((y_train, np.ones(len(generated_samples))))

        combined_samples, true_labels = shuffle(combined_samples, true_labels)
        predicted_labels = np.round_(np.nan_to_num(wgan.discriminator(combined_samples).numpy())).reshape(len(true_labels))
        g_measure = calculate_g_measure(true_labels=true_labels, predicted_labels=predicted_labels)

        return -g_measure

    search_space = {
        "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
        "generator_lr": hp.loguniform("generator_lr", np.log(0.0005), np.log(0.1)),
        "discriminator_lr": hp.loguniform("discriminator_lr", np.log(0.0005), np.log(0.1)),
        "generator_optimizer": hp.choice("generator_optimizer", [tf.keras.optimizers.Adadelta, 
                                                                 tf.keras.optimizers.Adagrad, 
                                                                 tf.keras.optimizers.Adam, 
                                                                 tf.keras.optimizers.Adamax, 
                                                                 tf.keras.optimizers.Nadam, 
                                                                 tf.keras.optimizers.RMSprop, 
                                                                 tf.keras.optimizers.SGD]),
        "discriminator_optimizer": hp.choice("discriminator_optimizer",
                                            [tf.keras.optimizers.Adadelta, 
                                             tf.keras.optimizers.Adagrad, 
                                             tf.keras.optimizers.Adam, 
                                             tf.keras.optimizers.Adamax, 
                                             tf.keras.optimizers.Nadam, 
                                             tf.keras.optimizers.RMSprop, 
                                             tf.keras.optimizers.SGD]),
        "generator_activation_fn": hp.choice("generator_activation_fn", ["relu", "leaky_relu", "tanh", "sigmoid"]),
        "discriminator_activation_fn": hp.choice("discriminator_activation_fn", ["relu", "leaky_relu", "tanh", "sigmoid"]),
        "epochs": hp.quniform("epochs", 5, 20, 1),
        "generator_layer_normalization": hp.choice("generator_layer_normalization", [True, False]),
        "discriminator_layer_normalization": hp.choice("discriminator_layer_normalization", [True, False])
    }

    start_time = time.time()

    trials = Trials()
    best_hyperparameters = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    best_hyperparameters = space_eval(search_space, best_hyperparameters)
    best_hyperparameters["generator_activation_fn"] = convert_activation_fn(
        best_hyperparameters["generator_activation_fn"]
    )
    best_hyperparameters["discriminator_activation_fn"] = convert_activation_fn(
        best_hyperparameters["discriminator_activation_fn"]
    )
    best_hyperparameters["epochs"] = int(best_hyperparameters["epochs"])

    best_g_measure = -trials.best_trial['result']['loss']
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best G-Measure:", best_g_measure)

    best_wgan = WGAN(X_train.shape[1], 1, 128, **best_hyperparameters)
    best_wgan.train_step(X_train)

    generated_samples = best_wgan.generator.predict(X_train)
    generated_labels = np.ones(len(generated_samples))

    X_train_balanced = np.concatenate((X_train, generated_samples))
    y_train_balanced = np.concatenate((y_train, generated_labels))

    X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced)
    
    return round(time.time() - start_time, 2), pd.DataFrame(X_train_balanced, columns=cols), pd.Series(y_train_balanced, name=tar)


# data_path = f"{os.getcwd()}\\data\\JavaScript_Vulnerability\\"
# datafiles = [f for f in os.listdir(data_path) if f.endswith("csv")]
# df = pd.read_csv(f"{data_path}\\{datafiles[0]}")
# drop_columns = ["name", "longname", "path", "full_repo_path", "line", "column", "endline", "endcolumn"]
# df = df.drop(drop_columns, axis=1)
# df = df.drop_duplicates()
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# X_train = shuffle(X_train)
# X_train.reset_index(inplace=True, drop=True)
# rt, X_train_new, y_train_new = DAZZLEOversampling(X_train=X_train, y_train=y_train)
# print(X_train_new)
# print(str(round(y_train_new.value_counts()[0] / y_train_new.value_counts()[1])))