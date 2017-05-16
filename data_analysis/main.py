import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor

import data_analysis.Learning
from data_analysis import Learning
from data_analysis.Graphs import prediction_hist
from data_analysis.Player import Stats

USE_SAVED_DATA = True
dataset_name = 'dataset.pkl'

if not (os.path.isfile(dataset_name) and USE_SAVED_DATA):
    Learning.generate_features(data_analysis.MATCH_LIST, dataset_name)
dataset = pickle.load(open(dataset_name, 'rb'))

# regr = GaussianProcessRegressor()
# regr = Ridge(alpha=0.1, normalize=True)
regr = MLPRegressor(hidden_layer_sizes=(100,))

# player_slot = 7
# stat = Stats.XPM
# match_id = data_analysis.TI_6_LAST_GAME_ID
# mean, std = Learning.predict_value(stat, regr, regr, match_id, dataset, player_slot=player_slot)
# actual_value = data_analysis.MATCHES[match_id]["players"][player_slot][stat.value]
# Graphs.plot_estimation(mean, std,
#                        "{} prediction for player {} in {}".format(stat.value,player_slot, data_analysis.TI_6_LAST_GAME_ID),
#                        actual_value)
# plt.savefig('{} for {} in game {}'.format(stat.value,player_slot,match_id),bbox_inches='tight')

dataset = Learning.new_format_to_old(dataset)

# targets = Learning.generate_targets(dataset['match_ids'], 0, Stats.GPM)
# print(np.mean(cross_val_score(regr, dataset['features'],targets,cv=10)))

for stat in Stats:
    results = Learning.test_stat(stat, regr, dataset)
    print()
    print("Stat: {}".format(stat.value))
    print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(results),
                                                      "Highest", max(results),
                                                      "Lowest", min(results)))

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
#     Graphs.raw_stat_hist(stat)
#     Graphs.error_stat_hist(stat, regr, dataset)
#     Graphs.prediction_hist(stat, regr, dataset)

# Graphs.prediction_hist(Stats.ASSISTS, regr, dataset)

# Graphs.fit_gaussian(Stats.GPM)
# Graphs.raw_trueskill_winrate(dataset)
# print("Graphs drawn")
# Graphs.raw_stat_hist(Stats.XPM)
fig = plt.figure(figsize=(12, 7.5))
# plt.plot([1.10, .869, .823, .808, .802, .801, .803, .819, .833, .838, .851, .852, .862], label=r"Validation loss")
# plt.plot([1.05, .858, .817, .775, .750, .725, .708, .700, .692, .683, .675, .675, .667], label=r"Training loss")
# plt.axis([0, 13, .65, 1.15])
# plt.ylabel('Loss')
# plt.xlabel('Co-ordinate descent iterations')
# plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
# plt.legend()

# H = np.array([[.1897, .2184, .2348, .2449, .2543],
#               [.1869, .2113, .2327, .2431, .2569],
#               [.1836, .2117, .2332, .2456, .2567],
#               [.1803, .2132, .2317, .2440, .2556],
#               [.1873, .2151, .2366, .2499, .2580],
#               [.1876, .2075, .2349, .2439, .2531],
#               ])  # added some commas and array creation code
#
# H = np.swapaxes(H, 0, 1)
# H = H[::-1]
#
# ax = fig.add_subplot(111)
# ax.set_title('Hyper-parameter search results')
# plt.imshow(H, cmap=plt.get_cmap('plasma'), interpolation='nearest', extent=[0.707, 45.25, 0.707, 22.63])
# ax.set_aspect('equal')
# plt.xscale('log', basex=2)
# plt.yscale('log', basey=2)
# plt.xlabel("Size of team skills")
# plt.ylabel("Size of player skills")
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_formatter(ScalarFormatter())
#
#
# def fmt(x, pos):
#     return '' if x == 0 else '{:.0f}'.format(x)
#
#
# ax.xaxis.set_major_formatter(FuncFormatter(fmt))
# ax.yaxis.set_major_formatter(FuncFormatter(fmt))
#
# plt.colorbar(orientation='vertical', label=r"$R^2$")

# high = [0.811, 0.805, 0.796, 0.793, 0.795, 0.796, 0.793, 0.792, 0.796, 0.787, 0.792, 0.802, 0.799, 0.801, 0.797, 0.798,
#         0.791, 0.81, 0.801, 0.798, 0.802]
# medium = [1.002, 1.000, 0.997, 0.882, 0.825, 0.817, 0.807, 0.803, 0.8, 0.799, 0.795, 0.797, 0.798, 0.795, 0.794, 0.796,
#           0.796, 0.795, 0.797, 0.796, 0.797]
# low =
# fit_gamma_and_gaussian(Stats.GPM)

# raw_stat_hist(Stats.GPM)
# plt.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     left='off',      # ticks along the bottom edge are off
#     right='off',         # ticks along the top edge are off
#     labelleft='off')
# plt.ylabel("Frequency")
# plt.xlabel("Variable value")

# plt.plot([0, 0, .181, .227, .265, .256, .229, .218, ])
# plt.xlabel("Number of months used")
# plt.ylim([0.15,0.3])
# plt.ylabel(r"Average $R^2$")
# plt.xlim([2, 7])

# raw_trueskill_winrate(dataset)
# prediction_hist(Stats.ASSISTS, regr, dataset)
plt.show()

# selected_features = Learning.choose_features(Stats.GPM, regr, dataset)
# print(selected_features)
