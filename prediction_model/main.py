import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_analysis.Graphs import density_hist
from prediction_model import SESSION, MATCH_LIST, BATCH_SIZE, PLAYERS_PER_TEAM, NUM_OF_TEAMS, RETRAIN, PLAYER_GAMES
from prediction_model.match_processing_model import player_results_split, player_to_results_param0, \
    player_to_results_param1, \
    loss as inference_loss, player_performance_estimate as match_performances, player_skills, player_results, \
    team_results, \
    player_to_team_nn, player_result_nn, team_result_nn
from prediction_model.skill_update_model import loss as update_loss, player_pre_skill, player_performance, \
    player_next_performance, post_skill
from prediction_model.utils import create_data_set, get_new_batch, store_player_performances, get_skill_batch, \
    update_player_skills, create_player_games, get_test_batch, split_list, make_mu_and_sigma, make_k_and_theta

test_result = player_results_split[0]
test2_result = player_to_results_param0[0] * player_to_results_param1[0]

inference_train_step = tf.train.AdamOptimizer().minimize(inference_loss)
update_train_step = tf.train.AdamOptimizer().minimize(update_loss)
init = tf.global_variables_initializer()
SESSION.run(init)

create_data_set()

# MATCH_LIST = [x for x in MATCH_LIST if BOSTON_MAJOR_LAST_GAME_ID > x > TI_5_LAST_GAME_ID]

train_list, test_list = split_list(MATCH_LIST, 0.9)

num_train_batches = int(len(train_list) / BATCH_SIZE)

create_player_games(train_list)

pass_num = 0
result = 0
alpha = .9
phase = True
counter = 0
min_update_loss = np.infty
stopping_counter = 0

file_name = "predictions.pkl"

if RETRAIN:
    while True:
        if stopping_counter > 5 and pass_num > 15:
            break
        if phase:
            batch = get_new_batch(counter, train_list, num_train_batches)
            _, loss_step, player_performances, test, test2 = SESSION.run(
                (inference_train_step, inference_loss, match_performances, test_result, test2_result),
                feed_dict={player_skills: batch["player_skills"],
                           player_results: batch["player_results"],
                           team_results: batch["team_results"]})
            result = (result * alpha + loss_step) / (1 + alpha)
            store_player_performances(batch["match_ids"], np.swapaxes(player_performances, 0, 1))
            if batch["switch"]:
                phase = not phase
                pass_num += 1
                print("Pass {}".format(pass_num))
                print("Inference loss:\t\t{:4d}".format(int(result)))
                result = 0
            counter += 1
        else:
            player_loss = []
            for player_id in PLAYER_GAMES.keys():
                batch = get_skill_batch(player_id)
                if len(batch["player_next_performance"]) == 0:
                    continue
                _, loss_step, player_skills_new = SESSION.run((
                    update_train_step, update_loss, post_skill),
                    feed_dict={player_pre_skill: batch["player_pre_skill"],
                               player_performance: batch["player_performance"],
                               player_next_performance: batch["player_next_performance"]})
                player_loss.append(loss_step)
                update_player_skills(player_id, batch["target_game"], player_skills_new)
            result = int(np.mean(player_loss))
            if result < min_update_loss:
                stopping_counter = 0
                min_update_loss = result
            else:
                stopping_counter += 1
            phase = not phase
            print("Skill update loss:\t{:4d}".format(int(result)))
            result = 0
        if np.math.isnan(result):
            print("Nan loss", file=sys.stderr)
            break

    num_test_batches = int(len(test_list) / BATCH_SIZE)
    predicted = []
    result = []
    error = []
    for seed in range(num_test_batches):
        batch = get_test_batch(seed, test_list)
        player_skills = np.array(batch["player_skills"])
        player_skills_split = tf.split(player_skills, PLAYERS_PER_TEAM * NUM_OF_TEAMS, axis=1)
        for i in range(len(player_skills_split)):
            player_skills_split[i] = tf.squeeze(player_skills_split[i], 1)
            player_skills_split[i] = tf.cast(player_skills_split[i], tf.float32)
            player_skills_split[i], _ = tf.split(player_skills_split[i], 2, axis=1)
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
        for i in range(PLAYERS_PER_TEAM * NUM_OF_TEAMS):
            if i < PLAYERS_PER_TEAM:
                team_skills = tf.concat([team0_skill, team1_skill], axis=1)
            else:
                team_skills = tf.concat([team1_skill, team0_skill], axis=1)
            player_to_result_input = tf.concat([player_skills_split[i], team_skills], axis=1)
            param0, param1 = make_mu_and_sigma(player_result_nn, player_to_result_input)
            predicted_player_result.append(param0)
        predicted_team_result = tf.sigmoid(team_result_nn(tf.concat([team0_skill, team1_skill], axis=1)))

        predicted_result = SESSION.run(predicted_player_result)
        predicted_result = np.swapaxes(predicted_result, 0, 1)
        for i in range(len(predicted_result)):
            for player in range(len(predicted_result[i])):
                predicted.append(predicted_result[i][player])
                result.append(batch["player_results"][i][player])
                error.append(predicted_result[i][player] - batch["player_results"][i][player])
    data = {"predicted": predicted, "result": result, "error": error}
    pickle.dump(data, open(file_name, "wb"))
else:
    data = pickle.load(open(file_name, "rb"))

predicted = np.swapaxes(data["predicted"], 0, 1)
error = np.swapaxes(data["error"], 0, 1)
result = np.swapaxes(data["result"], 0, 1)
stat = 1
density_hist(predicted[stat], label="predicted")
density_hist(result[stat], figure=False, label="result")
plt.legend()
density_hist(error[stat])
plt.show()

error = data["error"]
result = data["result"]
print()
print("Error std:    {}".format(np.std(error, 0)))
print("Original std: {}".format(np.std(result, 0)))
