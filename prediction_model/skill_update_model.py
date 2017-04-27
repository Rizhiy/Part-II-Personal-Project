import tensorflow as tf

from prediction_model import BATCH_SIZE, PLAYERS_PER_TEAM, NUM_OF_TEAMS, PLAYER_DIM
from prediction_model.utils import log_normal, make_mu_and_sigma, make_bigger_sql_nn

player_pre_skills = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PLAYERS_PER_TEAM * NUM_OF_TEAMS, PLAYER_DIM * 2))
player_performance = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PLAYERS_PER_TEAM * NUM_OF_TEAMS, PLAYER_DIM))
player_next_performance = tf.placeholder(tf.float32,
                                         shape=(BATCH_SIZE, PLAYERS_PER_TEAM * NUM_OF_TEAMS, PLAYER_DIM))

# Only one network here, since we are only doing forward pass
nn = make_bigger_sql_nn(PLAYER_DIM * 3, PLAYER_DIM * 2)

pre_skills_split = tf.split(player_pre_skills, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)
performance_split = tf.split(player_performance, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)
next_performance_split = tf.split(player_next_performance, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)

log_result = 0
post_skill = []
for i in range(len(pre_skills_split)):
    pre_skills_split[i] = tf.squeeze(pre_skills_split[i], 1)
    performance_split[i] = tf.squeeze(performance_split[i], 1)
    next_performance_split[i] = tf.squeeze(next_performance_split[i], 1)
    nn_input = tf.concat([pre_skills_split[i], performance_split[i]], axis=1)
    player_post_skill, sigma = make_mu_and_sigma(nn, nn_input)
    log_result += log_normal(next_performance_split[i], player_post_skill, sigma)
    post_skill.append(tf.concat((player_post_skill, sigma), axis=1))

loss = -tf.reduce_mean(log_result)
