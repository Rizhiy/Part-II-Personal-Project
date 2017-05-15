import copy
import random
import sys
from bisect import bisect
from enum import Enum

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, warnings
from sklearn.preprocessing import MinMaxScaler

from prediction_model import PLAYERS_PER_TEAM, PLAYER_SKILLS, MATCHES, MATCH_LIST, PLAYER_PERFORMANCES, \
    BATCH_SIZE, DEFAULT_PLAYER_SKILL, PLAYER_GAMES, GAMES_TO_CONSIDER, DEBUG


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
    error = -(tf.pow((x - mu) / sigma, 2) / 2 + min_log(sigma))
    return tf.reduce_sum(error, 1)


def log_entropy(sigma):
    e = min_log(sigma)
    return tf.reduce_sum(e, 1)


def log_bernoulli(y, p):
    p = tf.clip_by_value(p, 1e-8, 1 - 1e-8)
    result = y * min_log(p) + (1 - y) * min_log(1 - p)
    return tf.reduce_sum(result, 1)


def log_gamma(x, k, theta):
    error = (k - 1) * min_log(x) - (tf.lgamma(k) + k * min_log(theta) + x / theta)
    return tf.reduce_sum(error, 1)


def make_sql_nn(in_dim: int, out_dim: int, dropout: bool = False):
    p = keras.models.Sequential()
    p.add(Dense(units=int((in_dim + out_dim) / 2), input_dim=in_dim, kernel_initializer='random_normal'))
    p.add(Dense(units=out_dim, activation='linear', kernel_initializer='random_normal'))
    if dropout:
        p.add(Dropout(.2))
    return p


def make_bigger_sql_nn(in_dim: int, out_dim: int, dropout: bool = False):
    p = keras.models.Sequential()
    p.add(Dense(units=int((in_dim + out_dim)), input_dim=in_dim, activation='relu', kernel_initializer='random_normal'))
    p.add(Dense(units=int((in_dim + out_dim) / 2), activation='relu', kernel_initializer='random_normal'))
    p.add(Dense(units=out_dim, activation='linear', kernel_initializer='random_normal'))
    if dropout:
        p.add(Dropout(.2))
    return p


def make_smaller_sql_nn(in_dim: int, out_dim: int, dropout: bool = False):
    p = keras.models.Sequential()
    p.add(Dense(units=out_dim, input_dim=in_dim, activation='linear', kernel_initializer='random_normal'))
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
    players = []
    for player in match_data["players"]:
        players.append(player)
    return {
        "players": players,
        "match_id": match_id,
        "match_data": match_data
    }


def get_match_arrays(match_id):
    match_stats = get_player_data(match_id)
    radiant_ids = []
    dire_ids = []
    player_ids = []
    player_results = []
    for idx, player_stat in enumerate(match_stats["players"]):
        player_result = [player_stat[Stats.GPM.value], player_stat[Stats.XPM.value],
                         player_stat[Stats.CREEPS.value], player_stat[Stats.DENIES.value],
                         player_stat[Stats.KILLS.value], player_stat[Stats.DEATHS.value],
                         player_stat[Stats.ASSISTS.value], player_stat[Stats.LEVEL.value]]
        player_results.append(player_result)
        player_ids.append(player_stat["account_id"])
        if idx < PLAYERS_PER_TEAM:
            radiant_ids.append(player_stat["account_id"])
        else:
            dire_ids.append(player_stat["account_id"])
    if match_id not in PLAYER_SKILLS:
        skills = []
        for _ in range(10):
            skills.append(DEFAULT_PLAYER_SKILL)
        PLAYER_SKILLS[match_id] = skills
    if "radiant_win" not in match_stats["match_data"]:
        match_stats["match_data"]["radiant_win"] = True
    if match_stats["match_data"]["radiant_win"]:
        team_results = [1, 0]
    else:
        team_results = [0, 1]
    return {
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


DATASET = {"player_results": {},
           "team_results": {}}


def create_data_set():
    """
    Standardise player results
    """
    for match_id in MATCH_LIST:
        data = get_match_arrays(match_id)
        DATASET["player_results"][match_id] = data["player_results"]
        DATASET["team_results"][match_id] = data["team_results"]

    results = []
    for match_id in DATASET["player_results"]:
        for i in DATASET["player_results"][match_id]:
            results.append(i)
    # Remove outliers
    lower_clip = np.percentile(results, 1, 0)
    higher_clip = np.percentile(results, 99, 0)
    results = np.clip(results, lower_clip, higher_clip)
    scalar = MinMaxScaler(feature_range=(0, 1))
    scalar.fit(results)
    # temp fix, not fixing since not gonna use this in the future
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for match_id in DATASET["player_results"]:
        for idx, stats in enumerate(DATASET["player_results"][match_id]):
            DATASET["player_results"][match_id][idx] = scalar.transform(np.clip(stats, lower_clip, higher_clip))


def get_new_batch(seed, match_list, num_of_batches):
    batch = {"player_skills": [],
             "player_results": [],
             "team_results": [],
             "match_ids": [],
             "switch": False}
    if (seed + 1) * BATCH_SIZE % len(match_list) == 0:
        batch["switch"] = True
    seed = seed % num_of_batches
    for i in range(BATCH_SIZE):
        match_id = match_list[seed * BATCH_SIZE + i]
        batch["player_skills"].append(PLAYER_SKILLS[match_id])
        batch["player_results"].append(DATASET["player_results"][match_id])
        batch["team_results"].append(DATASET["team_results"][match_id])
        batch["match_ids"].append(match_id)
    return batch


def get_test_batch(seed, test_list):
    batch = {"player_skills": [],
             "player_results": [],
             "team_results": [],
             "match_ids": []}
    for i in range(BATCH_SIZE):
        match_id = test_list[seed * BATCH_SIZE + i]
        batch["player_results"].append(DATASET["player_results"][match_id])
        batch["team_results"].append(DATASET["team_results"][match_id])
        batch["match_ids"].append(match_id)
        skill_set = []
        for player_id in get_player_ids(match_id):
            skill_set.append(get_skill(player_id, match_id))
        batch["player_skills"].append(skill_set)
    return batch


def get_player_ids(match_id):
    player_ids = []
    players = get_match_data(match_id)["players"]
    for player in players:
        player_ids.append(int(player["account_id"]))
    return player_ids


def store_player_performances(match_ids, performances):
    for idx, match_id in enumerate(match_ids):
        PLAYER_PERFORMANCES[match_id] = performances[idx]


def get_skill_batch(player_id):
    batch = {"player_pre_skill": [],
             "player_performance": [],
             "player_next_performance": [],
             "target_game": []}
    player_games = PLAYER_GAMES[player_id]
    num_games = len(player_games["all"])
    for idx, match_id in enumerate(player_games["all"]):
        if idx - GAMES_TO_CONSIDER < 0:
            pre_skill = DEFAULT_PLAYER_SKILL
        else:
            skill_game = player_games["all"][idx - GAMES_TO_CONSIDER]
            slot = player_games[skill_game]["slot"]
            pre_skill = PLAYER_SKILLS[skill_game][slot]
        performances = []
        for i in range(GAMES_TO_CONSIDER):
            if idx - GAMES_TO_CONSIDER + i < 0:
                performances.append(DEFAULT_PLAYER_SKILL[:int(len(DEFAULT_PLAYER_SKILL) / 2)])
            else:
                game = player_games["all"][idx - GAMES_TO_CONSIDER + i]
                slot = player_games[game]["slot"]
                performances.append(PLAYER_PERFORMANCES[game][slot])
        if idx == num_games - 1:
            break
        else:
            slot = player_games[match_id]["slot"]
            next_performance = PLAYER_PERFORMANCES[match_id][slot][:int(len(DEFAULT_PLAYER_SKILL) / 2)]
        batch["player_pre_skill"].append(pre_skill)
        batch["player_performance"].append(performances)
        batch["player_next_performance"].append(next_performance)
        batch["target_game"].append(match_id)
    return batch


def update_player_skills(player_id: int, target_ids: list, skills: list):
    for idx, match_id in enumerate(target_ids):
        slot = PLAYER_GAMES[player_id][match_id]["slot"]
        PLAYER_SKILLS[match_id][slot] = skills[idx]
        PLAYER_SKILLS[match_id] = np.array(PLAYER_SKILLS[match_id])


def get_skill(player_id: int, match_id: int):
    if player_id not in PLAYER_GAMES:
        if DEBUG:
            print("Player not found: {}".format(player_id), file=sys.stderr)
        return DEFAULT_PLAYER_SKILL
    player_games = PLAYER_GAMES[player_id]
    match_list = player_games["all"]
    match_list.sort()
    idx = bisect(match_list, match_id)
    if idx == len(match_list):
        idx = -1
    target_id = match_list[idx]
    slot = player_games[target_id]["slot"]
    return PLAYER_SKILLS[target_id][slot]


def create_player_games(match_list: list):
    for match_id in match_list:
        for player in MATCHES[match_id]["players"]:
            player_id = player["account_id"]
            if player_id not in PLAYER_GAMES:
                PLAYER_GAMES[player_id] = {}
                PLAYER_GAMES[player_id]["all"] = []
            player_set = PLAYER_GAMES[player_id]
            if len(player_set["all"]) > 0:
                prev_match_id = player_set["all"][-1]
                player_set[prev_match_id]["next"] = match_id
                player_set[match_id] = {"prev": prev_match_id}
            else:
                player_set[match_id] = {"prev": None}

            slot = player["player_slot"]
            if slot > 10:
                slot = slot - 123  # - 128 + 5
            player_set[match_id]["slot"] = slot
            player_set[match_id]["next"] = None
            player_set["all"].append(match_id)


def split_list(match_list: list, ratio=0.8):
    match_list = copy.deepcopy(match_list)
    num = int((1 - ratio) * len(match_list))
    random.shuffle(match_list)
    test = match_list[:num]
    train = match_list[num:]
    test = test[len(test) % BATCH_SIZE:]
    train = train[len(train) % BATCH_SIZE:]
    test.sort()
    train.sort()
    return train, test


def split_into_chucks(match_list: list, num_of_chunks=5):
    match_list = copy.deepcopy(match_list)
    random.shuffle(match_list)
    size_of_chuck = int(len(match_list) / num_of_chunks)
    return [match_list[i:i + size_of_chuck] for i in range(0, len(match_list), size_of_chuck)]
