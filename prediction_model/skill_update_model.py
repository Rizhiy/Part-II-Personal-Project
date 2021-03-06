import tensorflow as tf

from prediction_model import PLAYER_DIM, GAMES_TO_CONSIDER
from prediction_model.utils import log_normal, make_mu_and_sigma, make_bigger_sql_nn

player_pre_skill = tf.placeholder(tf.float32, shape=(None, PLAYER_DIM * 2))
player_performance = tf.placeholder(tf.float32, shape=(None, GAMES_TO_CONSIDER, PLAYER_DIM))
player_next_performance = tf.placeholder(tf.float32, shape=(None, PLAYER_DIM))

batch_size = tf.shape(player_next_performance)[0]
epsilon = tf.random_normal((batch_size, PLAYER_DIM))

nn = make_bigger_sql_nn(GAMES_TO_CONSIDER * PLAYER_DIM + PLAYER_DIM * 2, PLAYER_DIM * 2)

player_performance_split = tf.split(player_performance, GAMES_TO_CONSIDER, axis=1)
for i in range(GAMES_TO_CONSIDER):
    player_performance_split[i] = tf.squeeze(player_performance_split[i], axis=1)
player_performance_new = tf.concat(player_performance_split, axis=1)

nn_input = tf.concat([player_pre_skill, player_performance_new], axis=1)
player_post_skill, sigma = make_mu_and_sigma(nn, nn_input)
log_result = log_normal(player_next_performance, player_post_skill, sigma)

post_skill = tf.concat([player_post_skill, sigma], axis=1)

loss = -tf.reduce_mean(log_result)
