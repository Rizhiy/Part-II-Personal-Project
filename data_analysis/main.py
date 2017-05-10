import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from sklearn.linear_model import Ridge

import data_analysis.Learning
from data_analysis import Learning

USE_SAVED_DATA = True
dataset_name = 'dataset.pkl'

if not (os.path.isfile(dataset_name) and USE_SAVED_DATA):
    Learning.generate_features(data_analysis.MATCH_LIST, dataset_name)
dataset = pickle.load(open(dataset_name, 'rb'))

regr = Ridge(alpha=0.1, normalize=True)
# regr = MLPRegressor(hidden_layer_sizes=(50,))

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

# Graphs.fit_gaussian(Stats.GPM)
# Graphs.fit_gamma_and_gaussian(Stats.GPM)
# Graphs.raw_trueskill_winrate(dataset)
# print("Graphs drawn")
# Graphs.raw_stat_hist(Stats.XPM)
# plt.plot([1.10, .869, .823, .808, .802, .801, .803, .819, .833, .838, .851, .852, .862], label="Validation loss")
# plt.plot([1.05, .858, .817, .775, .750, .725, .708, .700, .692, .683, .675, .675, .667], label="Training loss")
# plt.axis([0, 13, .65, 1.15])
# plt.ylabel('Loss')
# plt.xlabel('Co-ordinate descent iterations')
# plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
# plt.legend()

H = np.array([[.960, .835, .784, .777, .772, .769],
              [.896, .866, .772, .781, .807, .785],
              [.990, .949, .799, .791, .806, .799],
              [.971, .965, .815, .798, .798, .813],
              [1.01, .984, 1.02, .802, .809, .789],
              [1.02, 1.01, 1.01, 1.01, .812, .820],
              [1.03, 1.00, 1.02, .911, .926, .797],
              ])  # added some commas and array creation code

# fig = plt.figure(figsize=(6, 3.2))
fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('Hyper-parameter search results')
plt.imshow(H, cmap=plt.get_cmap('plasma_r'), interpolation='nearest', extent=[0.707, 45.25, 90.51, 0.707])
ax.set_aspect('equal')
plt.xscale('log', basex=2)
plt.yscale('log', basey=2)
plt.ylabel("Size of team skills")
plt.xlabel("Size of player skills")
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())


def fmt(x, pos):
    return '' if x == 0 else '{:.0f}'.format(x)


ax.xaxis.set_major_formatter(FuncFormatter(fmt))
ax.yaxis.set_major_formatter(FuncFormatter(fmt))

plt.colorbar(orientation='vertical').ax.invert_yaxis()
plt.show()

plt.show()

# selected_features = Learning.choose_features(Stats.GPM, regr, dataset)
# print(selected_features)
