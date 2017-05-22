import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers.core import K
from scipy.interpolate import spline

from prediction_model import SESSION, MATCH_LIST, RETRAIN, PLAYER_GAMES, BOSTON_MAJOR_LAST_GAME_ID, TI_6_LAST_GAME_ID, \
    DEBUG, PLAYER_PERFORMANCES, PLAYER_SKILLS, FEAR_ID, PLAYER_DIM, TEST_TEAM
from prediction_model.Graphs import density_hist
from prediction_model.system import test_system, train_system, draw_predictions
from prediction_model.utils import create_data_set, create_player_games, get_skill, split_list_3

init = tf.global_variables_initializer()
SESSION.run(init)

create_data_set()

match_list = [x for x in MATCH_LIST if BOSTON_MAJOR_LAST_GAME_ID > x > int(BOSTON_MAJOR_LAST_GAME_ID - 8 * 1e8)]

train_list, validation_list, test_list = split_list_3(match_list, 100)
if DEBUG > 0:
    if TI_6_LAST_GAME_ID in test_list:
        test_list.remove(TI_6_LAST_GAME_ID)
        train_list.append(TI_6_LAST_GAME_ID)
    if TI_6_LAST_GAME_ID in validation_list:
        validation_list.remove(TI_6_LAST_GAME_ID)
        train_list.append(TI_6_LAST_GAME_ID)
create_player_games(train_list)

file_name = "predictions.pkl"

if RETRAIN:
    train_system(train_list, validation_list)
    data = test_system(test_list)
    print("Test R-squared:       {:4.3f}".format(data["accuracy"]))
    pickle.dump(data, open(file_name, "wb"))
else:
    data = pickle.load(open(file_name, "rb"))
K.set_learning_phase(False)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)

if DEBUG > 0:
    print("Player skills at Ti 6:")
    print(PLAYER_SKILLS[TI_6_LAST_GAME_ID])
    print("Player performances at Ti 6:")
    print(PLAYER_PERFORMANCES[TI_6_LAST_GAME_ID])
if DEBUG > 1:
    predicted = np.swapaxes(data["predicted"], 0, 1)
    error = np.swapaxes(data["error"], 0, 1)
    result = np.swapaxes(data["result"], 0, 1)
    stat = 1
    density_hist(predicted[stat])
    density_hist(result[stat])
    # plt.legend()
    plt.figure()
    density_hist(error[stat])
    plt.show()

if DEBUG > 2:
    skills = []
    for match_id in PLAYER_GAMES[FEAR_ID]["all"]:
        skills.append(get_skill(FEAR_ID, match_id, True))
    skills = np.array(skills)

    l = len(skills[:, 0])
    xnew = np.linspace(0, l, 128)

    for i in range(PLAYER_DIM):
        plt.plot(xnew, spline(range(l), skills[:, i], xnew))
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                    labelleft='off')
    plt.xlabel("Time")
    plt.ylabel("Skill level")
    plt.show()

draw_predictions(TEST_TEAM)
plt.show()

error = data["error"]
result = data["result"]
print()
print("Error variance:    {}".format(repr(np.var(error, 0))))
print("Original variance: {}".format(repr(np.var(result, 0))))
print("R^2:               {}".format(repr([1] * 8 - np.var(error, 0) / np.var(result, 0))))
