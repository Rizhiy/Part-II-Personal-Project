import random
import sys

import numpy as np
import tensorflow as tf
from keras.layers.core import K

from prediction_model import SESSION, DEBUG, BATCH_SIZE, PLAYER_GAMES
from prediction_model.match_processing_model import loss as inference_loss, player_skills, player_results, team_results, \
    player_results_split, player_to_results_param0, player_to_results_param1, player_performance as match_performances, \
    player_performance_estimate
from prediction_model.p_model import predicted_player_result, predicted_player_skills
from prediction_model.skill_update_model import loss as update_loss, post_skill, player_post_skill, player_pre_skill, \
    player_performance, player_next_performance
from prediction_model.utils import get_new_batch, store_player_performances, get_test_batch, get_skill_batch, \
    update_player_skills

test_result = player_results_split[0]
test2_result = player_to_results_param0[0] * player_to_results_param1[0]

inference_train_step = tf.train.AdamOptimizer().minimize(inference_loss)
update_train_step = tf.train.AdamOptimizer().minimize(update_loss)

file_name = "predictions.pkl"


def train_prediction_model(train_list: list, validation_list: list):
    num_train_batches = int(len(train_list) / BATCH_SIZE)
    result = 1e5
    counter = random.randint(0, 1e5)
    pass_num = 0
    stopping_counter = 0
    validation_score = 0
    alpha = 0.9
    while True:
        batch = get_new_batch(counter, train_list, num_train_batches)
        _, loss_step, player_performances, performance_estimate, test, test2 = SESSION.run(
            (inference_train_step, inference_loss, match_performances, player_performance_estimate, test_result,
             test2_result),
            feed_dict={player_skills: batch["player_skills"],
                       player_results: batch["player_results"],
                       team_results: batch["team_results"]})
        result = (result * alpha + loss_step) / (1 + alpha)
        store_player_performances(batch["match_ids"], np.swapaxes(player_performances, 0, 1))
        if np.math.isnan(loss_step):
            print("Nan loss", file=sys.stderr)
            break
        if batch["switch"]:
            random.shuffle(train_list)
            pass_num += 1
            if DEBUG > 2:
                print("Inference loss:\t\t{:4d}".format(int(result)))
            if DEBUG > 3:
                print("Actual:      {}".format(test[0]))
                print("Inferred:    {}".format(test2[0]))
                print("Performance: {}".format(player_performances[0][0]))
                print("Prior:       {}".format(batch["player_skills"][0][0]))
            results = test_system(validation_list)
            if DEBUG > 1:
                print("Prediction training: {:4.3f}".format(results["accuracy"]))
            if results["accuracy"] > 0.05:
                if results["accuracy"] > validation_score:
                    validation_score = results["accuracy"]
                    stopping_counter = 0
                else:
                    stopping_counter += 1
            else:
                if loss_step < result:
                    stopping_counter = 0
                else:
                    stopping_counter += 1
            result = 0
            if stopping_counter > 2:
                break
        counter += 1


def train_update_model(validation_list: list):
    validation_score = 0
    stopping_counter = 0
    while True:
        player_loss = []
        ids = list(PLAYER_GAMES.keys())
        random.shuffle(ids)
        for player_id in ids:
            batch = get_skill_batch(player_id)
            if len(batch["player_next_performance"]) == 0:
                continue
            _, loss_step, player_skills_new, skills_mu = SESSION.run((
                update_train_step, update_loss, post_skill, player_post_skill),
                feed_dict={player_pre_skill: batch["player_pre_skill"],
                           player_performance: batch["player_performance"],
                           player_next_performance: batch["player_next_performance"]})
            update_player_skills(player_id, batch["target_game"], player_skills_new, skills_mu)
            player_loss.append(loss_step)
            if np.math.isnan(loss_step):
                print("Nan loss", file=sys.stderr)
                break
        if DEBUG > 1:
            print("Skill update loss:\t{:4d}".format(int(np.mean(player_loss))))
        results = test_system(validation_list)
        if results["accuracy"] > validation_score:
            validation_score = results["accuracy"]
            stopping_counter = 0
        else:
            stopping_counter += 1
        if stopping_counter > 2:
            break


def test_system(test_list: list):
    K.set_learning_phase(False)
    num_test_batches = int(len(test_list) / BATCH_SIZE)
    predicted = []
    result = []
    error = []
    for seed in range(num_test_batches):
        batch = get_test_batch(seed, test_list)
        predicted_result = SESSION.run(predicted_player_result, feed_dict={
            predicted_player_skills: batch["player_skills"]
        })
        predicted_result = np.swapaxes(predicted_result, 0, 1)
        for i in range(len(predicted_result)):
            for player in range(len(predicted_result[i])):
                predicted.append(predicted_result[i][player])
                result.append(batch["player_results"][i][player])
                error.append(batch["player_results"][i][player] - predicted_result[i][player])
    prediction = np.array(predicted)
    results = np.array(result)
    prediction_error = np.array(error)
    if DEBUG > 2:
        print("Error std:    {}".format(np.std(prediction_error, 0)))
        print("Original std: {}".format(np.std(results, 0)))
    accuracy = np.mean(1 - np.var(prediction_error, 0) / np.var(results, 0))
    K.set_learning_phase(True)
    data = {"predicted": prediction, "error": prediction_error, "result": results, "accuracy": accuracy}
    return data
