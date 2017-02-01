import sys
from enum import Enum

import numpy as np
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import data_analysis
from data_analysis import Match
from data_analysis.Player import Stats

from sys import maxsize
from copy import deepcopy


def generate_features(match_list, dataset_name):
    features = []
    match_ids = []
    match_list.sort()
    for counter, match_id in enumerate(match_list):
        print("\rPreparing dataset: " + ("{:.2f}".format((float(counter) / len(match_list)))), end='')
        sys.stdout.flush()
        results = Match.generate_feature_set(match_id)
        features.append(results["features"])
        match_ids.append(results["match_id"])
        Match.update_stats(match_id)
    print("\rPreparing dataset: Done")

    features = np.array(features)
    dataset = {'features': features, 'match_ids': match_ids}

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
        targets = generate_targets(dataset['match_ids'], player_slot, stat)
        scores.append(np.mean(cross_val_score(estimator, dataset['features'], targets, cv=10)))
    return scores


def test_duration(estimator, dataset):
    targets = []
    for match_id in dataset['match_ids']:
        targets.append(Match.get_match_data(match_id)["duration"])
    return cross_val_score(estimator, dataset["features"], targets, cv=10)


def test_winrate(estimator, dataset):
    targets = []
    for match_id in dataset["match_ids"]:
        targets.append(Match.get_match_data(match_id)["radiant_win"])
    return cross_val_score(estimator, dataset["features"], targets, cv=10)


def predict_stat(stat: Enum, estimator, dataset):
    results = []
    targets = []
    for player_slot in range(0, 10):
        targets = generate_targets(dataset['match_ids'], player_slot, stat)
    X_train, X_test, y_train, y_test = train_test_split(dataset["features"], targets)
    estimator.fit(X_train, y_train)
    for idx, match_features in enumerate(X_test):
        results.append(y_test[idx] - estimator.predict(match_features))
    results = [x + np.median(targets) for x in results]
    return results


def choose_features(stat: Enum, estimator, dataset):
    current_features = []
    current_best = -maxsize
    while True:
        result = find_next_feature(stat, estimator, dataset, current_features)
        if result["best_result"] < current_best:
            break
        current_features.append(result["best_feature"])
        print()
        print(current_features)
        print(result["best_result"])
    return current_features


def find_next_feature(stat: Enum, estimator, dataset, current_features_idx):
    all_features = dataset['features']

    current_features = select_current_features(dataset, current_features_idx)
    best_feature = None
    best_result = -maxsize
    for idx, feature in enumerate(all_features[0]):
        if idx in current_features_idx:
            continue
        print("\rChecking Feature {}: ".format(len(current_features_idx) + 1) + "{:.2f}".format(
            (float(idx) / len(all_features[0]))), end='')
        new_features = add_feature(dataset, current_features, idx)
        new_result = np.median(
            test_stat(stat, estimator, {'features': new_features, 'match_ids': dataset['match_ids']}))
        if new_result > best_result:
            best_feature = idx
            best_result = new_result
    return {
        "best_feature": best_feature,
        "best_result": best_result
    }


def select_current_features(dataset, current_features_idx):
    current_features = []
    for feature_set in dataset['features']:
        current_feature_set = []
        for idx, feature in enumerate(feature_set):
            if idx in current_features_idx:
                current_feature_set.append(feature)
        current_features.append(current_feature_set)
    return current_features


def add_feature(dataset, current_features, target_idx):
    new_features = deepcopy(current_features)
    for idx, feature_set in enumerate(dataset['features']):
        new_features[idx].append(feature_set[target_idx])
    return new_features
