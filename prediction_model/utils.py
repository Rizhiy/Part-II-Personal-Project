from enum import Enum

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, warnings
from sklearn.preprocessing import MinMaxScaler

from prediction_model import PLAYERS_PER_TEAM, PLAYERS, PLAYER_DIM, MATCHES, MATCH_LIST


class Stats(Enum):
    KILLS = "kills"
    DEATHS = "deaths"
    ASSISTS = "assists"
    LEVEL = "level"
    GPM = "gold_per_min"
    XPM = "xp_per_min"
    CREEPS = "last_hits"
    DENIES = "denies"
    # Those three values are missing from first half of the dataset,
    # TOWER_DMG = "tower_damage"
    # HERO_DMG = "hero_damage"
    # HEALING = "hero_healing"


def min_log(tensor):
    return tf.log(tf.clip_by_value(tensor, 1e-8, np.infty))


def clip_exp(tensor):
    return tf.exp(tf.clip_by_value(tensor, -5, 5))


def log_normal(x, mu, sigma):
    error = -(tf.pow((x - mu) / sigma, 2) / 2 + min_log(sigma) + min_log(2 * np.pi) / 2)
    return tf.reduce_sum(error, 1)


def log_bernoulli(y, p):
    p = tf.clip_by_value(p, 1e-8, 1 - 1e-8)
    result = y * min_log(p) + (1 - y) * min_log(1 - p)
    return tf.reduce_sum(result, 1)


def log_gamma(x, k, theta):
    # TODO: Check tf.lgamma
    error = (k - 1) * min_log(x) - (tf.lgamma(k) + k * min_log(theta) + x / theta)
    return tf.reduce_sum(error, 1)


def make_sql_nn(in_dim: int, out_dim: int, dropout: bool = False, first_activation='relu'):
    p = keras.models.Sequential()
    p.add(Dense(units=int((in_dim + out_dim) / 2), input_dim=in_dim, activation=first_activation,
                kernel_initializer='random_normal'))
    p.add(Dense(units=out_dim, activation='linear', kernel_initializer='random_normal'))
    if dropout:
        p.add(Dropout(.2))
    return p


def make_mu_and_sigma(nn, tensor):
    mu, log_sigma = tf.split(nn(tensor), num_or_size_splits=2, axis=1)
    # log_sigma = tf.clip_by_value(log_sigma, -5, 5)
    sigma = clip_exp(log_sigma)
    return mu, sigma


def make_k_and_theta(nn, tensor):
    log_k, log_theta = tf.split(nn(tensor), num_or_size_splits=2, axis=1)
    k = clip_exp(log_k)
    theta = clip_exp(log_theta)
    return k, theta


def get_match_data(match_id):
    return MATCHES[match_id]


def get_player_side(player_slot):
    """
    :param player_slot:
    :type player_slot: int
    :return: Returns true if radiant
    :rtype: bool
    """
    mask = 0b10000000
    return mask & player_slot == 0


def get_player_data(match_id: int) -> dict:
    """
    :param match_id: id of the required match
    :type match_id: int
    :return: dict with four values: "radiant_players", "dire_players", "match_id" and "match_data"
    :rtype: dict
    """
    match_data = get_match_data(match_id)
    radiant_players = []
    dire_players = []
    for player in match_data["players"]:
        if get_player_side(player["player_slot"]):
            radiant_players.append(player)
        else:
            dire_players.append(player)
    return {
        "radiant_players": radiant_players,
        "dire_players": dire_players,
        "match_id": match_id,
        "match_data": match_data
    }


def get_match_arrays(match_id):
    match_stats = get_player_data(match_id)
    radiant_ids = []
    dire_ids = []
    player_results = []
    players_stats = match_stats["radiant_players"] + match_stats["dire_players"]
    for idx, player_stat in enumerate(players_stats):
        player_result = [player_stat[Stats.GPM.value], player_stat[Stats.XPM.value],
                         player_stat[Stats.CREEPS.value], player_stat[Stats.DENIES.value],
                         player_stat[Stats.KILLS.value], player_stat[Stats.DEATHS.value],
                         player_stat[Stats.ASSISTS.value], player_stat[Stats.LEVEL.value]]
        player_results.append(player_result)
        if idx < PLAYERS_PER_TEAM:
            radiant_ids.append(player_stat["account_id"])
        else:
            dire_ids.append(player_stat["account_id"])
    player_skills = []
    for player_id in radiant_ids + dire_ids:
        if player_id not in PLAYERS:
            PLAYERS[player_id] = [0] * PLAYER_DIM + [1] * PLAYER_DIM
        player_skills.append(PLAYERS[player_id])
    if "radiant_win" not in match_stats["match_data"]:
        match_stats["match_data"]["radiant_win"] = True
    if match_stats["match_data"]["radiant_win"]:
        team_results = [1, 0]
    else:
        team_results = [0, 1]
    return {
        "player_skills": player_skills,
        "player_results": player_results,
        "team_results": team_results
    }


def get_batch(seed, batch_size):
    batch = {"player_skills": [],
             "player_results": [],
             "team_results": []}
    for i in range(batch_size):
        match_id = MATCH_LIST[(seed * batch_size + i) % len(MATCH_LIST)]
        data = get_match_arrays(match_id)
        batch["player_skills"].append(data["player_skills"])
        batch["player_results"].append(data["player_results"])
        batch["team_results"].append(data["team_results"])
    return batch


DATASET = {"player_skills": {},
           "player_results": {},
           "team_results": {}}


def create_data_set():
    """
    Standardise player results
    """
    for match_id in MATCH_LIST:
        data = get_match_arrays(match_id)
        DATASET["player_skills"][match_id] = data["player_skills"]
        DATASET["player_results"][match_id] = data["player_results"]
        DATASET["team_results"][match_id] = data["team_results"]

    results = []
    for match_id in DATASET["player_results"]:
        for i in DATASET["player_results"][match_id]:
            results.append(i)
    scalar = MinMaxScaler(feature_range=(0, 1))
    scalar.fit(results)
    # temp fix, not fixing since not gonna use this in the future
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for match_id in DATASET["player_results"]:
        for idx, stats in enumerate(DATASET["player_results"][match_id]):

            DATASET["player_results"][match_id][idx] = scalar.transform(stats)


def get_new_batch(seed, batch_size):
    batch = {"player_skills": [],
             "player_results": [],
             "team_results": []}
    for i in range(batch_size):
        match_id = MATCH_LIST[(seed * batch_size + i) % len(MATCH_LIST)]
        batch["player_skills"].append(DATASET["player_skills"][match_id])
        batch["player_results"].append(DATASET["player_results"][match_id])
        batch["team_results"].append(DATASET["team_results"][match_id])
    return batch
