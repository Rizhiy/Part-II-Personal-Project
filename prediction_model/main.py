import sys

import numpy as np
import tensorflow as tf

from prediction_model import SESSION, BATCH_SIZE
from prediction_model.match_processing_model import player_results_split, player_to_results_k, player_to_results_theta, \
    loss, player_performance, player_skills, player_results, team_results
from prediction_model.utils import create_data_set, get_new_batch, store_player_performances

test_result = player_results_split[0]
test2_result = player_to_results_k[0] * player_to_results_theta[0]

train_step = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()
SESSION.run(init)

create_data_set()

result = 0
alpha = .9
for i in range(int(1e6)):
    batch = get_new_batch(i, BATCH_SIZE)
    _, loss_step, player_skills_new, test, test2 = SESSION.run(
        (train_step, loss, player_performance, test_result, test2_result),
        feed_dict={player_skills: batch["player_skills"],
                   player_results: batch["player_results"],
                   team_results: batch["team_results"]})
    result = (result * alpha + loss_step) / (1 + alpha)
    store_player_performances(batch["match_ids"], np.swapaxes(player_skills_new, 0, 1))
    if np.math.isnan(loss_step):
        print("Nan loss", file=sys.stderr)
        break
    if i % 1000 == 0:
        print()
        print("iteration: {:6d}, loss: {:10.0f}".format(i, result))
        print("result:    {}".format(test[0]))
        print("inference: {}".format(test2[0]))
