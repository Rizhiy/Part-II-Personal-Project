import pickle
import keras
import numpy as np
import tensorflow as tf

from keras.layers import Activation, Dense

from tensorflow_tutorial import plot_images


def min_log(tensor):
    return tf.log(tensor + 1e-8)


def log_normal(x, mu, sigma):
    input = -(tf.pow((x - mu) / sigma, 2) / 2 + min_log(sigma) + min_log(2 * np.pi) / 2)
    return tf.reduce_sum(input, 1)


# What are we even trying to do here?

sess = tf.Session()

p = keras.models.Sequential()

dimZ = 50
dimX = 560
batch_size = 100

p.add(Dense(units=int((dimZ + dimX * 2) / 2), input_dim=dimZ))
p.add(Activation('softplus'))
p.add(Dense(units=dimX * 2))
p.add(Activation('linear'))

q = keras.models.Sequential()

q.add(Dense(units=int((dimX + dimZ * 2) / 2), input_dim=dimX))
q.add(Activation('softplus'))
q.add(Dense(units=dimZ * 2))
q.add(Activation('linear'))

epsilon = tf.random_normal((batch_size, dimZ))
x = tf.placeholder(tf.float32, shape=(batch_size, dimX))

mu_q, log_sigma_q = tf.split(q(x), num_or_size_splits=2, axis=1)
sigma_q = tf.exp(log_sigma_q)

z = mu_q + sigma_q * epsilon

mu_p, log_sigma_p = tf.split(p(z), num_or_size_splits=2, axis=1)
sigma_p = tf.exp(log_sigma_p)

log_q = log_normal(z, mu_q, sigma_q)
log_prior = log_normal(z, 0., 1.)
log_p = log_normal(x, mu_p, sigma_p)

log_result = log_p + log_prior - log_q
loss = -tf.reduce_mean(log_result)  # something is wrong since we can reduce mean below 0

data = pickle.load(open('freyfaces.pkl', 'rb'), encoding='latin1')
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
for _ in range(500):
    index = np.random.permutation(data.shape[0])[:100]
    batch = data[index]
    __, loss_step = sess.run((train_step, loss), feed_dict={x: batch})
    print(loss_step)

z = tf.random_normal((batch_size, dimZ))
mu_x, log_sigma_x = tf.split(p(z), num_or_size_splits=2, axis=1)
images = sess.run(mu_x)

plot_images(images,[28,20],'','faces')