import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pyplot as plt

import data_analysis.Learning
from data_analysis import Learning, Graphs
from data_analysis.Player import Stats

USE_SAVED_DATA = True
dataset_name = 'dataset.pkl'

if not (os.path.isfile(dataset_name) and USE_SAVED_DATA):
    Learning.generate_features(data_analysis.MATCH_LIST, dataset_name)
dataset = pickle.load(open(dataset_name, 'rb'))

regr = Ridge(alpha=0.1, normalize=True)

player_slot = 7
stat = Stats.XPM
match_id = data_analysis.TI_6_LAST_GAME_ID
mean, std = Learning.predict_value(stat, regr, regr, match_id, dataset, player_slot=player_slot)
actual_value = data_analysis.MATCHES[match_id]["players"][player_slot][stat.value]
Graphs.plot_estimation(mean, std,
                       "{} prediction for player {} in {}".format(stat.value,player_slot, data_analysis.TI_6_LAST_GAME_ID),
                       actual_value)
# plt.savefig('{} for {} in game {}'.format(stat.value,player_slot,match_id),bbox_inches='tight')

# dataset = Learning.new_format_to_old(dataset)


# for stat in Stats:
#     results = Learning.test_stat(stat, regr, dataset)
#     print()
#     print("Stat: {}".format(stat.value))
#     print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(results),
#                                                       "Highest", max(results),
#                                                       "Lowest", min(results)))

# duration_results = Learning.test_duration(regr, dataset)
# print()
# print("Stat: {}".format("Duration"))
# print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(duration_results),
#                                                   "Highest", max(duration_results),
#                                                   "Lowest", min(duration_results)))

# clf = GaussianNB()
# winrate_results = Learning.test_winrate(clf, dataset)
# print()
# print("Stat: {}".format("Winrate"))
# print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(winrate_results),
#                                                   "Highest", max(winrate_results),
#                                                   "Lowest", min(winrate_results)))

# for stat in Stats:
# Graphs.raw_stat_hist(stat)
# Graphs.error_stat_hist(stat, regr, dataset)
# Graphs.prediction_hist(stat, regr, dataset)

# Graphs.raw_trueskill_winrate(dataset)
# print("Graphs drawn")
plt.show()

# selected_features = Learning.choose_features(Stats.GPM, regr, dataset)
# print(selected_features)
