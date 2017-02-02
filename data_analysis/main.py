import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR

import matplotlib.pyplot as plt

import data_analysis.Learning
from data_analysis import Learning, Graphs
from data_analysis.Player import Stats

USE_SAVED_DATA = True
dataset_name = 'dataset.pkl'

if not (os.path.isfile(dataset_name) and USE_SAVED_DATA):
    Learning.generate_features(data_analysis.MATCH_LIST, dataset_name)
dataset = pickle.load(open(dataset_name, 'rb'))

regr = Ridge()
# for stat in Stats:
#     results = Learning.test_stat(stat, regr, dataset)
#
#     print()
#     print("Stat: {}".format(stat.value))
#     print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(results),
#                                                       "Highest", max(results),
#                                                       "Lowest", min(results)))
#
# duration_results = Learning.test_duration(regr, dataset)
# print()
# print("Stat: {}".format("Duration"))
# print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(duration_results),
#                                                   "Highest", max(duration_results),
#                                                   "Lowest", min(duration_results)))
#
# clf = MLPClassifier(hidden_layer_sizes=(2000, 1000, 500, 100))
# winrate_results = Learning.test_winrate(clf, dataset)
# print()
# print("Stat: {}".format("Winrate"))
# print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(winrate_results),
#                                                   "Highest", max(winrate_results),
#                                                   "Lowest", min(winrate_results)))

# Graphs.raw_stat_hist(Stats.GPM)
# Graphs.raw_stat_hist(Stats.XPM)
# Graphs.raw_stat_hist(Stats.LEVEL)
# Graphs.raw_stat_hist(Stats.KILLS)
#
# Graphs.error_stat_hist(Stats.GPM,regr,dataset)
# Graphs.error_stat_hist(Stats.XPM,regr,dataset)
# Graphs.error_stat_hist(Stats.LEVEL,regr,dataset)
# Graphs.error_stat_hist(Stats.KILLS,regr,dataset)

# Graphs.raw_trueskill_winrate(dataset)
# print("Graphs drawn")
# plt.show()

# selected_features = Learning.choose_features(Stats.GPM, regr, dataset)
# print(selected_features)
