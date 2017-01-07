import json
import random

from data_analysis import Player
import data_analysis


def generate_feature_set(match_id, player_id=0):
    match_data = get_match_data(match_id)
    radiant_players = []
    dire_players = []
    for player in match_data["players"]:
        if (Player.get_player_side(player["player_slot"])):
            radiant_players.append(player["account_id"])
        else:
            dire_players.append(player["account_id"])
        if player['account_id'] == player_id:
            desired_player = player
    players = radiant_players + dire_players
    if player_id in players:
        players.remove(player_id)
        players = [player_id] + players
    player_objects = []
    for player in players:
        player_objects.append(Player.Player(player, match_id - 1))
    features = []
    for player in player_objects:
        features += player.get_features()
    return {
        "features": features,
        "target": desired_player['gold_per_min']
    }


def get_match_data(match_id):
    return data_analysis.MATCHES[match_id]


def get_random_set(match_list, test_ratio=0.1, selection_ratio=0.5):
    """
    Selects a random subset from given set and splits it into train and test parts numerically (everything in test is greater than everything in train).
    :param match_list:
    :type match_list: list
    :param test_ratio:
    :type test_ratio: float
    :param selection_ratio:
    :type selection_ratio: float
    :return: dictionary with two entries: "train_set" and "test_set"
    :rtype:dict
    """
    random.shuffle(match_list)
    result = match_list[0:int(len(match_list) * selection_ratio)]
    result.sort()
    train = result[0:int(len(result) * (1 - test_ratio))]
    test = result[int(len(result) * (1 - test_ratio)):-1]
    return {
        "train_set": train,
        "test_set": test
    }


def get_player_ids(match_id):
    """

    :param match_id:
    :type match_id: int
    :return: 3 entries: "radiant_players", "dire_players" and "all_players"
    :rtype:dict
    """
    match_data = data_analysis.MATCHES[match_id]
    dire_players = []
    radiant_players = []
    all_players = []
    for player in match_data["players"]:
        player_id = player["account_id"]
        if Player.get_player_side(player["player_slot"]):
            radiant_players.append(player_id)
        else:
            dire_players.append(player_id)
        all_players.append(player_id)
    return {
        "radiant_players": radiant_players,
        "dire_players": dire_players,
        "all_players": all_players
    }
