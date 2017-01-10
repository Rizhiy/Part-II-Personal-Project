import os
import sys
import numpy as np
import data_analysis
import pickle
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

from trueskill import Rating
from trueskill import expose

# temp fix, not fixing since not gonna use this in the future
import warnings

USE_SAVED_DATA = False

warnings.filterwarnings("ignore", category=DeprecationWarning)

dataset_name = 'dataset.pkl'

if os.path.isfile(dataset_name) and USE_SAVED_DATA:
    dataset = pickle.load(open(dataset_name, 'rb'))
else:
    X = []
    y = []
    match_list = data_analysis.MATCH_LIST
    match_list.sort()
    for counter, match_id in enumerate(match_list):
    for counter, match_id in enumerate(match_list):
        print("\rPreparing dataset: " + ("{:.2f}".format((float(counter) / len(match_list)))), end='')
        sys.stdout.flush()
        results = Match.generate_feature_set(match_id)
        X.append(results["features"])
        y.append(results["target"])
        Match.update_stats(match_id)
    print("\rPreparing dataset: Done")

    X = np.array(X)
    dataset = {'X': X, 'y': y}

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)

clf = MLPClassifier(hidden_layer_sizes=(200, 200, 200))
# X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['y'], test_size=0.1, random_state=42)
# regr.fit(X_train,y_train)
print(np.mean(cross_val_score(clf, dataset['X'], dataset['y'], cv=10)))

# dataset = Match.get_random_set(data_analysis.MATCH_LIST, selection_ratio=0.9)

# for idx, match_id in enumerate(dataset["train_set"]):
#     print("\rRanking players: " + ("{:.2f}".format(float(idx) / len(dataset["train_set"]))), end='')
#     sys.stdout.flush()
#     Match.update_stats(match_id)
# print("\rRanking players: Done")
#
# error = []
# for idx, match_id in enumerate(dataset["test_set"]):
#     print("\rEvaluating: " + ("{:.2f}".format(float(idx) / len(dataset["test_set"]))), end='')
#     radiant_probability = Match.predict_outcome(match_id)
#     predicted_outcome = radiant_probability > 0.5
#     if predicted_outcome == Match.radiant_win(match_id):
#         error.append(0.5 - abs(0.5 - predicted_outcome))
#     else:
#         error.append(0.5 + abs(0.5 - predicted_outcome))
#     Match.update_stats(match_id)
# print("\rEvaluating: Done")
#
# print("Mean Error: {}".format(np.mean(error)))

# players = list(data_analysis.PLAYERS.values())
# leaderboard = sorted(players, key=lambda x: expose(x.winrate), reverse=True)
#
# for player in leaderboard:
#     print(player.short_string())
#
# print("\n{:>12}:\n".format("DENDI") + str(data_analysis.PLAYERS[data_analysis.DENDI_ID]))
# print("\n{:>12}:\n".format("PUPPY") + str(data_analysis.PLAYERS[data_analysis.PUPPEY_ID]))
# print("\n{:>12}:\n".format("MISERY") + str(data_analysis.PLAYERS[data_analysis.MISERY_ID]))
# print("\n{:>12}:\n".format("RESOLUTION") + str(data_analysis.PLAYERS[data_analysis.RESOLUTION_ID]))
# print("\n{:>12}:\n".format("MIRACLE") + str(data_analysis.PLAYERS[data_analysis.MIRACLE_ID]))
# print("\n{:>12}:\n".format("S4") + str(data_analysis.PLAYERS[data_analysis.S4_ID]))
