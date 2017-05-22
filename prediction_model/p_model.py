import tensorflow as tf

from prediction_model import PLAYERS_PER_TEAM, NUM_OF_TEAMS, PLAYER_DIM
from prediction_model.match_processing_model import player_to_team_nn, player_result_nn, team_result_nn
from prediction_model.utils import make_mu_and_sigma, make_k_and_theta

predicted_player_skills = tf.placeholder(tf.float32, shape=(None, PLAYERS_PER_TEAM * NUM_OF_TEAMS, PLAYER_DIM))

player_skills_split = tf.split(predicted_player_skills, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)
for i in range(len(player_skills_split)):
    player_skills_split[i] = tf.squeeze(player_skills_split[i], 1)
    player_skills_split[i] = tf.cast(player_skills_split[i], tf.float32)
team0_input = []
team1_input = []
for i in range(PLAYERS_PER_TEAM):
    team0_input.append(player_skills_split[i])
    team1_input.append(player_skills_split[PLAYERS_PER_TEAM + i])
team0_input = tf.concat(team0_input, axis=1)
team1_input = tf.concat(team1_input, axis=1)
team0_skill, _ = make_mu_and_sigma(player_to_team_nn, team0_input)
team1_skill, _ = make_mu_and_sigma(player_to_team_nn, team1_input)
predicted_player_result = []
predicted_param0 = []
predicted_param1 = []
for i in range(PLAYERS_PER_TEAM * NUM_OF_TEAMS):
    if i < PLAYERS_PER_TEAM:
        team_skills = tf.concat([team0_skill, team1_skill], axis=1)
    else:
        team_skills = tf.concat([team1_skill, team0_skill], axis=1)
    player_to_result_input = tf.concat([player_skills_split[i], team_skills], axis=1)
    param0, param1 = make_k_and_theta(player_result_nn, player_to_result_input)
    predicted_param0.append(param0)
    predicted_param1.append(param1)
    predicted_player_result.append(param0 * param1)
predicted_team_result = tf.sigmoid(team_result_nn(tf.concat([team0_skill, team1_skill], axis=1)))
