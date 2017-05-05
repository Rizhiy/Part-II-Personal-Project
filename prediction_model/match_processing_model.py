import tensorflow as tf

from prediction_model import PLAYER_DIM, PLAYERS_PER_TEAM, NUM_OF_TEAMS, PLAYER_RESULT_DIM, TEAM_RESULTS_DIM, TEAM_DIM, \
    BATCH_SIZE
from prediction_model.utils import make_mu_and_sigma, make_sql_nn, log_normal, log_bernoulli, entropy, \
    make_k_and_theta, log_gamma

# Do we really need float32? How much faster will float16 be? Same for ints
player_skills = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PLAYERS_PER_TEAM * NUM_OF_TEAMS, PLAYER_DIM * 2))
player_results = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PLAYERS_PER_TEAM * NUM_OF_TEAMS, PLAYER_RESULT_DIM))
team_results = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TEAM_RESULTS_DIM,))

team0_epsilon = tf.random_normal((BATCH_SIZE, TEAM_DIM,))
team1_epsilon = tf.random_normal((BATCH_SIZE, TEAM_DIM,))
player_epsilons = []
for i in range(PLAYERS_PER_TEAM * NUM_OF_TEAMS):
    player_epsilons.append(tf.random_normal((BATCH_SIZE, PLAYER_DIM,)))

# Upwards pass
result_to_team_nn = make_sql_nn(PLAYER_RESULT_DIM * PLAYERS_PER_TEAM * NUM_OF_TEAMS + TEAM_RESULTS_DIM, TEAM_DIM * 2)
player_skill_nn = make_sql_nn(PLAYER_RESULT_DIM + TEAM_DIM, PLAYER_DIM * 2)

# Downwards pass
player_to_team_nn = make_sql_nn(PLAYER_DIM * PLAYERS_PER_TEAM, TEAM_DIM * 2)
player_result_nn = make_sql_nn(PLAYER_DIM + TEAM_DIM * NUM_OF_TEAMS, PLAYER_RESULT_DIM * 2)
team_result_nn = make_sql_nn(TEAM_DIM * NUM_OF_TEAMS, TEAM_RESULTS_DIM)

player_results_split = tf.split(player_results, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)
player_skills_split = tf.split(player_skills, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)

for i in range(len(player_results_split)):
    player_results_split[i] = tf.squeeze(player_results_split[i], 1)
    player_skills_split[i] = tf.squeeze(player_skills_split[i], 1)

team0_performance_input = tf.concat([tf.reshape(player_results, [BATCH_SIZE, -1]), team_results], axis=1)
team0_performance_mu, team0_performance_sigma = make_mu_and_sigma(result_to_team_nn, team0_performance_input)
team0_performance = team0_performance_mu + team0_performance_sigma * team0_epsilon

# TODO: we need to swap first two numbers in team_results rather than reversing it in the future
team1_performance_input = tf.concat(
    [tf.reshape(tf.reverse(player_results, axis=(1,)), [BATCH_SIZE, -1]), tf.reverse(team_results, axis=(1,))], axis=1)
team1_performance_mu, team1_performance_sigma = make_mu_and_sigma(result_to_team_nn, team1_performance_input)
team1_performance = team1_performance_mu + team1_performance_sigma * team1_epsilon

player_performance = []
player_performance_sigma = []
player_performance_estimate = []
for i in range(PLAYERS_PER_TEAM * NUM_OF_TEAMS):
    if i < PLAYERS_PER_TEAM:
        team_performance = team0_performance
    else:
        team_performance = team1_performance

    player_performance_input = tf.concat([player_results_split[i], team_performance], axis=1)
    mu, sigma = make_mu_and_sigma(player_skill_nn, player_performance_input)
    player_performance_estimate.append(tf.concat([mu, sigma], axis=1))
    player_performance_sigma.append(sigma)
    player_performance.append(mu + sigma * player_epsilons[i])

player_to_team0_input = []
player_to_team1_input = []
for i in range(PLAYERS_PER_TEAM):
    player_to_team0_input.append(player_performance[i])
    player_to_team1_input.append(player_performance[PLAYERS_PER_TEAM + i])

player_to_team0_input = tf.concat(player_to_team0_input, axis=1)
player_to_team0_mu, player_to_team0_sigma = make_mu_and_sigma(player_to_team_nn, player_to_team0_input)

player_to_team1_input = tf.concat(player_to_team1_input, axis=1)
player_to_team1_mu, player_to_team1_sigma = make_mu_and_sigma(player_to_team_nn, player_to_team1_input)

player_to_results_param0 = []
player_to_results_param1 = []
for i in range(PLAYERS_PER_TEAM * NUM_OF_TEAMS):
    if i < PLAYERS_PER_TEAM:
        team_performances = tf.concat([team0_performance, team1_performance], axis=1)
    else:
        team_performances = tf.concat([team1_performance, team0_performance], axis=1)
    player_to_result_input = tf.concat([player_performance[i], team_performances], axis=1)
    param0, param1 = make_k_and_theta(player_result_nn, player_to_result_input)
    player_to_results_param0.append(param0)
    player_to_results_param1.append(param1)

team_to_result = tf.sigmoid(team_result_nn(tf.concat([team0_performance, team1_performance], axis=1)))

log_result = 0
for i in range(PLAYERS_PER_TEAM * NUM_OF_TEAMS):
    log_result += log_gamma(player_results_split[i], player_to_results_param0[i], player_to_results_param1[i])
    mu, sigma = tf.split(player_skills_split[i], 2, axis=1)
    log_result += log_normal(player_performance[i], mu, sigma)
    log_result -= entropy(player_performance_sigma[i])

log_result += log_bernoulli(team_results, team_to_result)

log_result += log_normal(team0_performance, player_to_team0_mu, player_to_team0_sigma)
log_result += log_normal(team1_performance, player_to_team1_mu, player_to_team1_sigma)

log_result -= entropy(team0_performance_sigma)
log_result -= entropy(team1_performance_sigma)

loss = -tf.reduce_mean(log_result)
