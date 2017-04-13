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


def make_sql_nn(in_dim, out_dim, dropout=False):
    p = keras.models.Sequential()
    p.add(Dense(units=int((in_dim + out_dim) / 2), input_dim=in_dim, activation='relu'))
    p.add(Dense(units=out_dim, activation='linear'))
    if dropout:
        p.add(Dropout(.2))
    return p


def make_mu_and_sigma(nn, tensor):
    mu, log_sigma = tf.split(nn(tensor), num_or_size_splits=2, axis=1)
    sigma = tf.exp(log_sigma)
    return mu, sigma


K.set_learning_phase(True)
sess = tf.Session()

dimZ_2 = 50
dimZ_1 = 50
dimZ_0 = 1000
dimX = 560

batch_size = 100

p_X = make_sql_nn(dimZ_0, dimX * 2, True)
p_0 = make_sql_nn(dimZ_1 + dimZ_2, dimZ_0 * 2)

q_0 = make_sql_nn(dimX, dimZ_0 * 2)
q_1 = make_sql_nn(dimZ_0, dimZ_1 * 2)
q_2 = make_sql_nn(dimZ_0, dimZ_2 * 2)

epsilon0 = tf.random_normal((batch_size, dimZ_0))
epsilon1 = tf.random_normal((batch_size, dimZ_1))
epsilon2 = tf.random_normal((batch_size, dimZ_1))
x = tf.placeholder(tf.float32, shape=(batch_size, dimX))

mu_q0, sigma_q0 = make_mu_and_sigma(q_0, x)

z_0 = mu_q0 + sigma_q0 * epsilon0

mu_pX, sigma_pX = make_mu_and_sigma(p_X, z_0)

mu_q1, sigma_q1 = make_mu_and_sigma(q_1, z_0)
mu_q2, sigma_q2 = make_mu_and_sigma(q_2, z_0)

z_1 = mu_q1 + sigma_q1 * epsilon1
z_2 = mu_q2 + sigma_q2 * epsilon2

z_1_2 = tf.concat([z_1, z_2], axis=1)

mu_p0, sigma_p0 = make_mu_and_sigma(p_0, z_1_2)

log_prior1 = log_normal(z_1, 0., 1.)
log_prior2 = log_normal(z_2, 0., 1.)
log_pX = log_normal(x, mu_pX, sigma_pX)
log_p0 = log_normal(z_0, mu_p0, sigma_p0)
log_q0 = log_normal(z_0, mu_q0, sigma_q0)
log_q1 = log_normal(z_1, mu_q1, sigma_q1)
log_q2 = log_normal(z_2, mu_q2, sigma_q2)

log_result = log_prior1 + log_prior2 + log_pX + log_p0 - log_q0 - log_q1 - log_q2
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
        z_2 = tf.random_normal((batch_size, dimZ_2))
        z_1_2 = tf.concat([z_1, z_2], axis=1)
        mu_0, _ = make_mu_and_sigma(p_0, z_1_2)
        mu_x, _ = make_mu_and_sigma(p_X, mu_0)
        images = sess.run(mu_x)
        plot_images(images, [28, 20], 'faces/', '4_' + str(i+1))
        K.set_learning_phase(True)
    index = np.random.permutation(data.shape[0])[:100]
    batch = data[index]
    _, loss_step = sess.run((train_step, loss), feed_dict={x: batch})
    if i % 100 == 0:
        print("iteration: {:5d}, score: {:5.0f}".format(i, -loss_step))
