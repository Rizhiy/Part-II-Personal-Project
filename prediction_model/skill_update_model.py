import tensorflow as tf

from prediction_model import PLAYER_DIM, GAMES_TO_CONSIDER
from prediction_model.utils import log_normal, make_mu_and_sigma, make_bigger_sql_nn, entropy

player_pre_skill = tf.placeholder(tf.float32, shape=(None, PLAYER_DIM * 2))
player_performance = tf.placeholder(tf.float32, shape=(None, GAMES_TO_CONSIDER, PLAYER_DIM))
player_next_performance = tf.placeholder(tf.float32, shape=(None, PLAYER_DIM))
player_next_performance_estimate = tf.placeholder(tf.float32, shape=(None, PLAYER_DIM * 2))

batch_size = tf.shape(player_next_performance)[0]
epsilon = tf.random_normal((batch_size, PLAYER_DIM))

# Only one network here, since we are only doing forward pass
q = make_bigger_sql_nn(PLAYER_DIM + GAMES_TO_CONSIDER * PLAYER_DIM * 2, PLAYER_DIM * 2)
p = make_bigger_sql_nn(PLAYER_DIM, PLAYER_DIM * 2)

player_performance_split = tf.split(player_performance, GAMES_TO_CONSIDER, axis=1)
for i in range(GAMES_TO_CONSIDER):
    player_performance_split[i] = tf.squeeze(player_performance_split[i], axis=1)
player_performance_new = tf.concat(player_performance_split, axis=1)

log_result = 0
q_input = tf.concat([player_pre_skill, player_performance_new], axis=1)
post_skill_mu, post_skill_sigma = make_mu_and_sigma(q, q_input)
log_result -= entropy(post_skill_sigma)
performance_sample = post_skill_mu + post_skill_sigma * epsilon

player_post_skill, sigma = make_mu_and_sigma(p, performance_sample)
log_result += log_normal(player_next_performance, player_post_skill, sigma)

post_skill = tf.concat([player_post_skill, sigma], axis=1)

loss = -tf.reduce_mean(log_result)
