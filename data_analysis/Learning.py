import sys
from collections import OrderedDict
from enum import Enum

import numpy as np
import pickle

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import data_analysis
from data_analysis import Match

from sys import maxsize
from copy import deepcopy


def generate_features(match_list, dataset_name):
    dataset = OrderedDict()
    match_list.sort()
    for counter, match_id in enumerate(match_list):
        print("\rPreparing dataset: " + ("{:.2f}".format((float(counter) / len(match_list)))), end='')
        sys.stdout.flush()
        results = Match.generate_feature_set(match_id)
        dataset[results['match_id']] = results['features']
        Match.update_stats(match_id)
    print("\rPreparing dataset: Done")

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)


def calculate_features(match_list, dataset_name):
    dataset = []
    match_list.sort()
    for counter, match_id in enumerate(match_list):
        print("\rPreparing dataset: " + ("{:.2f}".format((float(counter) / len(match_list)))), end='')
        sys.stdout.flush()
        results = Match.get_player_data(match_id)
        dataset.append(results)
        Match.update_stats(match_id)
    print("\rPreparing dataset: Done")

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)


def generate_targets(match_list, player_slot, stat):
    targets = []
    for match_id in match_list:
        match_data = data_analysis.MATCHES[match_id]
        targets.append(match_data["players"][player_slot][stat.value])
    return targets


def generate_targets_new(match_dict: OrderedDict, player_slot, stat):
    targets = {}
    for match_id, _ in match_dict.items():
        match_data = data_analysis.MATCHES[match_id]
        targets[match_id] = match_data["players"][player_slot][stat.value]
    return targets


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
    for player_slot in range(0, 10):
        targets = generate_targets(dataset['match_ids'], player_slot, stat)
        X_train, X_test, y_train, y_test = train_test_split(dataset["features"], targets)
        estimator.fit(X_train, y_train)
        for idx, match_features in enumerate(X_test):
            results.append(estimator.predict(match_features))
    return results


def test_error(stat: Enum, estimator, dataset: OrderedDict):
    results = []
    for player_slot in range(0, 10):
        errors = calculate_sqr_error_whole_set(stat, estimator, dataset, player_slot)
        targets = list(errors.values())
        features = features_from_dataset(dataset, list(errors.keys()))
        results.append(np.mean(cross_val_score(estimator, features, targets, cv=10)))
    return results


def calculate_sqr_error_whole_set(stat: Enum, estimator, dataset: OrderedDict, player_slot=0) -> OrderedDict:
    results = OrderedDict()
    sets = Match.split_into_sets(list(dataset.keys()))
    for idx, set in enumerate(sets):
        temp_dataset = OrderedDict(deepcopy(dataset))
        test_set = OrderedDict()

        for match_id in set:
            features = temp_dataset.pop(match_id, None)
            if features:
                test_set[match_id] = features

        y_test = generate_targets_new(test_set, player_slot, stat)
        y_train = generate_targets(temp_dataset.keys(), player_slot, stat)
        estimator.fit(list(temp_dataset.values()), y_train)
        for match_id, match_features in test_set.items():
            result = y_test[match_id] - estimator.predict(match_features)
            results[match_id] = abs(result * result)
    return results


def predict_value(stat: Enum, estimator, error_estimator, match_id, dataset: OrderedDict, player_slot=0, trained=False,
                  error_trained=False):
    dataset = deepcopy(dataset)
    predicition_features = dataset.pop(match_id)
    estimator = deepcopy(estimator)
    error_estimator = deepcopy(error_estimator)
    if not trained:
        targets = generate_targets(list(dataset.keys()), player_slot, stat)
        features = list(dataset.values())
        estimator.fit(features, targets)
    if not error_trained:
        errors = calculate_sqr_error_whole_set(stat, estimator, dataset, player_slot)
        targets = list(errors.values())
        features = features_from_dataset(dataset, list(errors.keys()))
        error_estimator.fit(features, targets)
    predicted_value = estimator.predict(predicition_features)
    predicted_std = np.sqrt(error_estimator.predict(predicition_features))[0]
    return predicted_value, predicted_std


# TODO: A lot of repetition here, need to refactor.
def calculate_error(stat: Enum, estimator, dataset):
    results = []
    for player_slot in range(0, 10):
        targets = generate_targets(dataset['match_ids'], player_slot, stat)
        X_train, X_test, y_train, y_test = train_test_split(dataset["features"], targets)
        estimator.fit(X_train, y_train)
        for idx, match_features in enumerate(X_test):
            results.append(y_test[idx] - estimator.predict(match_features))
    return results


def choose_features(stat: Enum, estimator, dataset):
    current_features = []
    current_best = -maxsize
    while True:
        result = find_next_feature(stat, estimator, dataset, current_features)
        num_features = len(current_features)
        if num_features > 10 and result["best_result"] / current_best < (num_features + 1) / num_features:
            break
        current_features.append(result["best_feature"])
        print()
        print("Selected features: " + str(current_features))
        print("Score: " + str(result["best_result"]))
    return current_features


def find_next_feature(stat: Enum, estimator, dataset, current_features_idx):
    all_features = dataset['features']

    current_features = select_current_features(dataset, current_features_idx)
    best_feature = None
    best_result = -maxsize
    for idx, feature in enumerate(all_features[0]):
        if idx in current_features_idx:
            continue
        print("\rSelecting Feature {}: ".format(len(current_features_idx) + 1) + "{:.2f}".format(
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


def new_format_to_old(dataset):
    features = []
    match_ids = []
    for match_id, feature_list in dataset.items():
        features.append(feature_list)
        match_ids.append(match_id)
    return {'features': features, 'match_ids': match_ids}


def dict_to_two_list(dictionary: dict):
    keys = []
    values = []
    for k, v in dictionary.items():
        keys.append(k)
        values.append(v)
    return {'keys': keys, 'values': values}


def features_from_dataset(dataset: dict, match_ids: list):
    features = []
    for match_id in match_ids:
        features.append(dataset[match_id])
    return features
