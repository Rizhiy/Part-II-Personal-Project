import pickle

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.layers.core import K

from tensorflow_tutorial import plot_images


def min_log(tensor):
    return tf.log(tensor + 1e-8)


def log_normal(x, mu, sigma):
    input = -(tf.pow((x - mu) / sigma, 2) / 2 + min_log(sigma) + min_log(2 * np.pi) / 2)
    return tf.reduce_sum(input, 1)


K.set_learning_phase(True)
sess = tf.Session()

p_1 = keras.models.Sequential()
p_0 = keras.models.Sequential()

q_1 = keras.models.Sequential()
q_0 = keras.models.Sequential()

dimZ_1 = 50
dimZ_0 = 50
dimX = 560

batch_size = 100

p_1.add(Dense(units=int((dimZ_1 + dimZ_0 * 2) / 2), input_dim=dimZ_1, activation='relu'))
p_1.add(Dense(units=dimZ_0 * 2, activation='linear'))
# p_1.add(Dropout(.2))  # Do I need dropout on every NN or just on the last one?

q_1.add(Dense(units=int((dimZ_0 + dimZ_1 * 2) / 2), input_dim=dimZ_0, activation='relu'))
q_1.add(Dense(units=dimZ_1 * 2, activation='linear'))
# q_1.add(Dropout(.2))

p_0.add(Dense(units=int((dimZ_0 + dimX * 2) / 2), input_dim=dimZ_0, activation='relu'))
p_0.add(Dense(units=dimX * 2, activation='linear'))
p_0.add(Dropout(.2))

q_0.add(Dense(units=int((dimX + dimZ_0 * 2) / 2), input_dim=dimX, activation='relu'))
q_0.add(Dense(units=dimZ_0 * 2, activation='linear'))
# q_0.add(Dropout(.2))

epsilon0 = tf.random_normal((batch_size, dimZ_0))
epsilon1 = tf.random_normal((batch_size, dimZ_1))
x = tf.placeholder(tf.float32, shape=(batch_size, dimX))

mu_q0, log_sigma_q0 = tf.split(q_0(x), num_or_size_splits=2, axis=1)
sigma_q0 = tf.exp(log_sigma_q0)

z_0 = mu_q0 + sigma_q0 * epsilon0

mu_p0, log_sigma_p0 = tf.split(p_0(z_0), num_or_size_splits=2, axis=1)
sigma_p0 = tf.exp(log_sigma_p0)

mu_q1, log_sigma_q1 = tf.split(q_1(z_0), num_or_size_splits=2, axis=1)
sigma_q1 = tf.exp(log_sigma_q1)

z_1 = mu_q1 + sigma_q1 * epsilon1

mu_p1, log_sigma_p1 = tf.split(p_1(z_1), num_or_size_splits=2, axis=1)
sigma_p1 = tf.exp(log_sigma_p1)

log_prior = log_normal(z_1, 0., 1.)
log_p0 = log_normal(x, mu_p0, sigma_p0)
log_q0 = log_normal(z_0, mu_q0, sigma_q0)
log_p1 = log_normal(z_0, mu_p1, sigma_p1)
log_q1 = log_normal(z_1, mu_q1, sigma_q1)

log_result = log_p0 + log_p1 + log_prior - log_q0 - log_q1
loss = -tf.reduce_mean(log_result)

data = pickle.load(open('freyfaces.pkl', 'rb'), encoding='latin1')
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# How to properly compute loss with dropout? Or does it even matter?
for i in range(50000):
    if (i + 1) % 1e4 == 0:
        K.set_learning_phase(False)
        z_1 = tf.random_normal((batch_size, dimZ_1))
        mu_1, _ = tf.split(p_1(z_1), num_or_size_splits=2, axis=1)
        mu_x, _ = tf.split(p_0(mu_1), num_or_size_splits=2, axis=1)
        images = sess.run(mu_x)
        plot_images(images, [28, 20], 'faces/', '3_' + str(i+1))
        K.set_learning_phase(True)
    index = np.random.permutation(data.shape[0])[:100]
    batch = data[index]
    if i % 100 == 0:
        _, loss_step = sess.run((train_step, loss), feed_dict={x: batch})
        print("iteration: {:5d}, score: {:5.0f}".format(i, -loss_step))
    else:
        _, loss_step = sess.run((train_step, loss), feed_dict={x: batch})
