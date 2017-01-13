import sys
import numpy as np
import pickle

from sklearn.cross_validation import cross_val_score

import data_analysis
from data_analysis import Match
from data_analysis.Player import Stats


def generate_features(match_list, dataset_name):
    X = []
    y = []
    match_list.sort()
    for counter, match_id in enumerate(match_list):
        print("\rPreparing dataset: " + ("{:.2f}".format((float(counter) / len(match_list)))), end='')
        sys.stdout.flush()
        results = Match.generate_feature_set(match_id)
        X.append(results["features"])
        y.append(results["match_id"])
        Match.update_stats(match_id)
    print("\rPreparing dataset: Done")

    X = np.array(X)
    dataset = {'X': X, 'y': y}

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)


def generate_targets(match_list, player_slot, stat):
    features = []
    for match_id in match_list:
        match_data = data_analysis.MATCHES[match_id]
        if stat == Stats.KDA:
            kills = match_data["players"][player_slot][Stats.KILLS.value]
            assists = match_data["players"][player_slot][Stats.ASSISTS.value]
            deaths = match_data["players"][player_slot][Stats.DEATHS.value]
            if not deaths:
                deaths = 1  # can't divide by zero
            features.append((kills + assists) / deaths)
        else:
            features.append(match_data["players"][player_slot][stat.value])
    return features


def test_stat(stat, estimator, dataset):
    scores = []
    for player_slot in range(0, 10):
        targets = generate_targets(dataset['y'], player_slot, stat)
        scores.append(np.mean(cross_val_score(estimator, dataset['X'], targets, cv=10)))
    return scores
