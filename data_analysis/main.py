import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

import data_analysis.Learning
from data_analysis import Learning

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
#     Graphs.raw_stat_hist(stat)
#     Graphs.error_stat_hist(stat, regr, dataset)
#     Graphs.prediction_hist(stat, regr, dataset)

# Graphs.prediction_hist(Stats.ASSISTS, regr, dataset)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)

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

# H = np.array([[.071, .170, .250, .246, .218],
#               [.074, .171, .262, .241, .220],
#               [.080, .167, .264, .250, .250],
#               [.091, .171, .263, .254, .246],
#               [.086, .173, .244, .250, .252],
#               [.090, .173, .256, .252, .253],
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
# plt.colorbar(orientation='vertical', label=r"Average $R^2$")

# high = [
#     [0, 0.218, 0.232, 0.254, 0.239, 0.247, 0.259, 0.259, 0.239, 0.265, 0.255],
#     [0, 0.236, 0.254, 0.239, 0.228, 0.236, 0.235, 0.25, 0.243, 0.244, 0.241],
#     [0, 0.232, 0.244, 0.231, 0.231, 0.232, 0.23, 0.234, 0.232, 0.245, 0.235],
#     [0, 0.229, 0.233, 0.238, 0.223, 0.223, 0.22, 0.227, 0.226, 0.219, 0.218],
#     [0, 0.236, 0.24, 0.24, 0.242, 0.229, 0.229, 0.24, 0.239, 0.241, 0.241]
# ]
# high = np.mean(high, 0)
# medium = [
#     [0, 0.088, 0.24, 0.236, 0.242, 0.24, 0.243, 0.243, 0.239, 0.239, 0.245],
#     [0, 0.158, 0.24, 0.239, 0.237, 0.242, 0.238, 0.242, 0.242, 0.238, 0.24],
#     [0, 0.13, 0.261, 0.272, 0.272, 0.269, 0.27, 0.27, 0.269, 0.272, 0.275],
#     [0, 0.002, 0.253, 0.26, 0.26, 0.258, 0.257, 0.256, 0.253, 0.258, 0.259],
#     [0, 0.117, 0.26, 0.261, 0.266, 0.268, 0.263, 0.264, 0.26, 0.261, 0.266]
#
# ]
# medium = np.mean(medium, 0)
# low = [
#     [0, -0.003, 0.01, 0.201, 0.253, 0.267, 0.258, 0.254, 0.259, 0.26, 0.259],
#     [0, -0.01, -0.001, 0.173, 0.218, 0.23, 0.231, 0.238, 0.226, 0.22, 0.223],
#     [0, -0.005, -0.003, 0.101, 0.249, 0.272, 0.277, 0.272, 0.273, 0.27, 0.267],
#     [0, -0.004, -0.001, 0.194, 0.251, 0.259, 0.264, 0.268, 0.267, 0.267, 0.264],
#     [0, -0.007, -0.001, 0.146, 0.236, 0.249, 0.251, 0.256, 0.253, 0.266, 0.262]
# ]
# low = np.mean(low, 0)

# plt.plot(high, label=r"High (5e-3)")
# plt.plot(medium, label=r"Medium (1e-3)")
# plt.plot(low, label=r"Low (5e-4)")
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
# plt.legend(loc="lower right")
plt.show()

error_var = [[0.026, 0.024, 0.03, 0.027, 0.037, 0.048, 0.057, 0.04],
             [0.029, 0.026, 0.031, 0.025, 0.035, 0.048, 0.054, 0.035],
             [0.027, 0.026, 0.032, 0.025, 0.038, 0.047, 0.054, 0.038],
             [0.027, 0.024, 0.033, 0.025, 0.039, 0.048, 0.056, 0.036],
             [0.029, 0.025, 0.037, 0.025, 0.038, 0.048, 0.055, 0.039],
             [0.028, 0.025, 0.032, 0.025, 0.037, 0.046, 0.057, 0.039],
             [0.027, 0.025, 0.03, 0.024, 0.039, 0.048, 0.055, 0.035],
             [0.027, 0.023, 0.036, 0.028, 0.035, 0.046, 0.051, 0.035],
             [0.026, 0.024, 0.029, 0.024, 0.04, 0.05, 0.058, 0.036],
             [0.028, 0.025, 0.035, 0.026, 0.043, 0.051, 0.057, 0.04],
             ]

org_var = [[0.054, 0.044, 0.05, 0.038, 0.045, 0.049, 0.058, 0.049],
           [0.058, 0.046, 0.053, 0.035, 0.043, 0.05, 0.056, 0.048],
           [0.055, 0.046, 0.052, 0.036, 0.046, 0.049, 0.055, 0.048],
           [0.055, 0.044, 0.055, 0.037, 0.048, 0.051, 0.058, 0.047],
           [0.057, 0.045, 0.059, 0.037, 0.047, 0.05, 0.056, 0.05],
           [0.056, 0.044, 0.052, 0.036, 0.045, 0.048, 0.059, 0.049],
           [0.057, 0.045, 0.051, 0.034, 0.048, 0.051, 0.056, 0.046],
           [0.057, 0.043, 0.058, 0.04, 0.043, 0.048, 0.053, 0.046],
           [0.053, 0.043, 0.05, 0.034, 0.048, 0.052, 0.06, 0.046],
           [0.057, 0.042, 0.058, 0.038, 0.052, 0.053, 0.059, 0.051],
           ]

R2 = [[0.516, 0.45, 0.405, 0.302, 0.16, 0.029, 0.02, 0.183],
      [0.506, 0.448, 0.409, 0.294, 0.184, 0.053, 0.042, 0.269],
      [0.503, 0.437, 0.371, 0.3, 0.183, 0.047, 0.027, 0.209],
      [0.517, 0.451, 0.4, 0.316, 0.191, 0.061, 0.043, 0.244],
      [0.502, 0.437, 0.372, 0.328, 0.2, 0.05, 0.032, 0.208],
      [0.508, 0.426, 0.394, 0.313, 0.166, 0.046, 0.034, 0.197],
      [0.519, 0.438, 0.414, 0.31, 0.189, 0.056, 0.029, 0.246],
      [0.522, 0.462, 0.382, 0.3, 0.175, 0.04, 0.037, 0.235],
      [0.499, 0.442, 0.416, 0.305, 0.164, 0.05, 0.032, 0.229],
      [0.507, 0.418, 0.405, 0.316, 0.179, 0.04, 0.035, 0.214],
      ]
print(np.mean(org_var, 0))
print(np.mean(error_var, 0))
print(np.mean(R2, 0))
print(np.std(R2, 0, ddof=1))


# selected_features = Learning.choose_features(Stats.GPM, regr, dataset)
# print(selected_features)
