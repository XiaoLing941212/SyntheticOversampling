"""
This file contains necessary WGAN and related functions

Reference: https://github.com/LyudaK/msc_thesis_imblearn/blob/main/ImbLearn.ipynb
Modified by: Xiao Ling, xling4@ncsu.edu
"""

import tensorflow as tf


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
    critic_optimizer.appply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    tf.group(*(var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in critic.trainable_variables))

    return wasserstein, gen_loss, critic_loss_real, critic_loss_fake