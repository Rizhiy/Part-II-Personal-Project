import tensorflow as tf
import keras
import numpy as np
from scipy.io import loadmat


def log_normal(x, mu, sigma):
    input = -(tf.pow((x - mu) / sigma, 2) / 2 + tf.log(sigma) + tf.log(2 * np.pi) / 2)
    return tf.reduce_sum(input, 1)


sess = tf.Session()
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# c = a + b
# d = c * 3
#
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# for i in range(1000):
#     sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
# print(sess.run([W, b]))
from keras.layers import Activation, Dense

p = keras.models.Sequential()

dimZ = 50
dimX = 560
batch_size = 100

p.add(Dense(units=int((dimZ + dimX * 2) / 2), input_dim=dimZ))
p.add(Activation('relu'))
p.add(Dense(units=dimX * 2))
p.add(Activation('linear'))

q = keras.models.Sequential()

q.add(Dense(units=int((dimX + dimZ * 2) / 2), input_dim=dimX))
q.add(Activation('relu'))
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
loss = -tf.reduce_mean(log_result)

data = loadmat("frey_rawface.mat")
data = np.swapaxes(data['ff'],0,1)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
for __ in range(1000):
    index = np.random.permutation(data.shape[0])[:100]
    batch = data[index]
    _, loss_step = sess.run((train_step, loss), feed_dict={x: batch})
    print(_,loss_step)
