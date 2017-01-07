import os
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from data_analysis import Match, Player
import data_analysis
import pickle
from trueskill import Rating
from trueskill import rate

# temp fix, not fixing since not gonna use this in the future
# import warnings
#
# USE_SAVED_DATA = False
#
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# classifier_name = 'classifier.pkl'
# dataset_name = 'dataset.pkl'
#
# if os.path.isfile(dataset_name) and USE_SAVED_DATA:
#     dataset = pickle.load(open(dataset_name, 'rb'))
# else:
#     X = []
#     y = []
#     counter = 0
#     player_id = data_analysis.DENDI_ID
#     match_list = Player.Player(player_id).get_all_matches()
#     for match_id in match_list:
#         counter += 1
#         print("\rPreparing dataset: " + ('%.2f' % (float(counter) / len(match_list))), end='')
#         sys.stdout.flush()
#         results = Match.generate_feature_set(match_id, player_id)
#         X.append(results["features"])
#         y.append(results["target"])
#     print("\rPreparing dataset: Done")
#
#     X = np.array(X)
#     dataset = {'X': X, 'y': y}
#
#     with open(dataset_name, 'wb') as output:
#         pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
#
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# clf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# # X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['y'], test_size=0.1, random_state=42)
# # regr.fit(X_train,y_train)
# print(np.mean(cross_val_score(clf, dataset['X'], dataset['y'], cv=10)))

dataset = data_analysis.MATCH_LIST

for idx, match_id in enumerate(dataset):
    player_ids = Match.get_player_ids(match_id)
    radiant_ratings = []
    dire_ratings = []
    for player_id in player_ids["all_players"]:
        if (not player_id in data_analysis.PLAYERS):
            data_analysis.PLAYERS[player_id] = Player.Player(player_id)
    for player_id in player_ids["radiant_players"]:
        radiant_ratings.append(data_analysis.PLAYERS[player_id].winrate)
    for player_id in player_ids["dire_players"]:
        dire_ratings.append(data_analysis.PLAYERS[player_id].winrate)
    if Match.get_match_data(match_id)["radiant_win"]:
        ranks = [1, 0]
    else:
        ranks = [0, 1]
    new_radiant_ratings, new_dire_ratings = rate([radiant_ratings, dire_ratings], ranks=ranks)
    for idx, player_id in enumerate(player_ids["radiant_players"]):
        data_analysis.PLAYERS[player_id].winrate = new_radiant_ratings[idx]
        data_analysis.PLAYERS[player_id].total_games += 1
    for idx, player_id in enumerate(player_ids["dire_players"]):
        data_analysis.PLAYERS[player_id].winrate = new_dire_ratings[idx]
        data_analysis.PLAYERS[player_id].total_games += 1

players = list(data_analysis.PLAYERS.values())
players.sort(key=lambda x: x.winrate.mu - 3 * x.winrate.sigma)

# for player in players:
#     print(player)

print("DENDI:\t" + str(data_analysis.PLAYERS[data_analysis.DENDI_ID]))
print("PUPPY:\t" + str(data_analysis.PLAYERS[data_analysis.PUPPEY_ID]))
print("MISERY:\t" + str(data_analysis.PLAYERS[data_analysis.MISERY_ID]))
print("RESOLUTION:\t" + str(data_analysis.PLAYERS[data_analysis.RESOLUTION_ID]))
print("MIRACLE:\t" + str(data_analysis.PLAYERS[data_analysis.MIRACLE_ID]))
print("S4:\t" + str(data_analysis.PLAYERS[data_analysis.S4_ID]))
