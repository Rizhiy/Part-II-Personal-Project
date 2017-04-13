import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout

from data_analysis.Match import get_match_stats
from data_analysis.Player import Stats
from prediction_model import PLAYERS_PER_TEAM, PLAYERS, PLAYER_DIM


def min_log(tensor):
    return tf.log(tensor + 1e-8)


def log_normal(x, mu, sigma):
    error = -(tf.pow((x - mu) / sigma, 2) / 2 + min_log(sigma) + min_log(2 * np.pi) / 2)
    return tf.reduce_sum(error, 1)


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


def get_match_arrays(match_id):
    match_stats = get_match_stats(match_id)
    radiant_ids = []
    dire_ids = []
    player_results = []
    players_stats = match_stats["radiant_players"] + match_stats["dire_players"]
    for idx, player_stat in enumerate(players_stats):
        player_results.append(player_stat[Stats.KILLS])
        player_results.append(player_stat[Stats.DEATHS])
        player_results.append(player_stat[Stats.ASSISTS])
        player_results.append(player_stat[Stats.GPM])
        player_results.append(player_stat[Stats.XPM])
        player_results.append(player_stat[Stats.LEVEL])
        player_results.append(player_stat[Stats.CREEPS])
        player_results.append(player_stat[Stats.DENIES])
        if idx < PLAYERS_PER_TEAM:
            radiant_ids = player_stat["account_id"]
        else:
            dire_ids = player_stat["account_id"]
    player_skills = []
    for player_id in radiant_ids + dire_ids:
        if player_id not in PLAYERS:
            PLAYERS[player_id] = [0] * PLAYER_DIM
        player_skills += PLAYERS[player_id]
    if match_stats["match_data"]["radiant_win"]:
        team_results = [1,0]
    else:
        team_results = [0,1]
    return {
        "player_skills": player_skills,
        "player_results": player_results,
        "team_results": team_results
    }
