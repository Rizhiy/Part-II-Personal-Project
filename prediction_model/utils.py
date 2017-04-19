from enum import Enum

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout

from prediction_model import PLAYERS_PER_TEAM, PLAYERS, PLAYER_DIM, MATCHES


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
    return tf.log(tensor + 1e-8)


def log_normal(x, mu, sigma):
    error = -(tf.pow((x - mu) / sigma, 2) / 2 + min_log(sigma) + min_log(2 * np.pi) / 2)
    error = tf.squeeze(error)
    return tf.reduce_sum(error, 0)


def make_sql_nn(in_dim: int, out_dim: int, dropout: bool = False, first_activation='relu'):
    p = keras.models.Sequential()
    p.add(Dense(units=int((in_dim + out_dim) / 2), input_dim=in_dim, activation=first_activation))
    p.add(Dense(units=out_dim, activation='linear'))
    if dropout:
        p.add(Dropout(.2))
    return p


def make_mu_and_sigma(nn, tensor):
    mu, log_sigma = tf.split(nn(tensor), num_or_size_splits=2, axis=1)
    sigma = tf.exp(log_sigma)
    return mu, sigma


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
        player_result = [player_stat[Stats.KILLS.value], player_stat[Stats.DEATHS.value],
                         player_stat[Stats.ASSISTS.value], player_stat[Stats.LEVEL.value],
                         player_stat[Stats.GPM.value], player_stat[Stats.XPM.value],
                         player_stat[Stats.CREEPS.value], player_stat[Stats.DENIES.value]]
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
    if match_stats["match_data"]["radiant_win"]:
        team_results = [1, 0]
    else:
        team_results = [0, 1]
    return {
        "player_skills": [player_skills],
        "player_results": [player_results],
        "team_results": [team_results]
    }
