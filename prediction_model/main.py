import sys

import numpy as np
import tensorflow as tf

from prediction_model import SESSION
from prediction_model.match_processing_model import player_results_split, player_to_results_k, player_to_results_theta, \
    loss as inference_loss, player_performance as match_performances, player_skills, player_results, team_results
from prediction_model.skill_update_model import loss as update_loss, player_pre_skills, player_performance, \
    player_next_performance, post_skill
from prediction_model.utils import create_data_set, get_new_batch, store_player_performances, get_skill_batch, \
    update_player_skills

test_result = player_results_split[0]
test2_result = player_to_results_k[0] * player_to_results_theta[0]

inference_train_step = tf.train.AdamOptimizer().minimize(inference_loss)
update_train_step = tf.train.AdamOptimizer().minimize(update_loss)
init = tf.global_variables_initializer()
SESSION.run(init)

create_data_set()

pass_num = 0
result = 0
alpha = .9
phase = True
counter = 0
while True:
    if phase:
        batch = get_new_batch(counter)
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
    else:
        batch = get_skill_batch(counter)
        _, loss_step, player_skills_new = SESSION.run((
            update_train_step, update_loss, post_skill),
            feed_dict={player_pre_skills: batch["player_pre_skills"],
                       player_performance: batch["player_performances"],
                       player_next_performance: batch["player_next_performances"]})
        result = (result * alpha + loss_step) / (1 + alpha)
        update_player_skills(batch["match_ids"], np.swapaxes(player_skills_new, 0, 1))
        if batch["switch"]:
            phase = not phase
            print("Skill update loss:\t{:4d}".format(int(result)))
            result = 0
    if np.math.isnan(loss_step):
        print("Nan loss", file=sys.stderr)
        break
    counter += 1
