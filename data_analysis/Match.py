import math
import random

import trueskill
from trueskill import rate
from trueskill.backends import cdf

import data_analysis
from data_analysis import Player

from copy import deepcopy


def generate_feature_set(match_id):
    match_data = get_match_data(match_id)
    radiant_players = []
    dire_players = []
    for player in match_data["players"]:
        if get_player_side(player["player_slot"]):
            radiant_players.append(player["account_id"])
        else:
            dire_players.append(player["account_id"])
    players_ids = radiant_players + dire_players
    features = []
    for player_id in players_ids:
        features += Player.get_player(player_id).get_features()
    return {
        "features": features,
        "match_id": match_id
    }


def get_player_data(match_id):
    match_data = get_match_data(match_id)
    radiant_players = []
    dire_players = []
    for player in match_data["players"]:
        player_data = deepcopy(data_analysis.PLAYERS[player["account_id"]])
        if get_player_side(player["player_slot"]):
            radiant_players.append(player_data)
        else:
            dire_players.append(player_data)
    return {
        "radiant_players": radiant_players,
        "dire_players": dire_players,
        "match_id": match_id,
        "match_data": match_data
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


def split_into_sets(match_list, num_sets=10):
    random.shuffle(match_list)
    set_size = len(match_list) // num_sets
    sets = []
    for set_num in range(0, num_sets):
        first_element = set_num * set_size
        if set_num == num_sets - 1:
            last_element = len(match_list)
        else:
            last_element = (set_num + 1) * set_size
        sets.append(match_list[first_element:last_element])
    return sets


def get_player_ids(match_id):
    """

    :param match_id:
    :type match_id: int
    :return: 3 entries: "radiant_players", "dire_players" and "all_players"
    :rtype: dict
    """
    match_data = data_analysis.MATCHES[match_id]
    dire_players = []
    radiant_players = []
    all_players = []
    for player in match_data["players"]:
        player_id = player["account_id"]
        if get_player_side(player["player_slot"]):
            radiant_players.append(player_id)
        else:
            dire_players.append(player_id)
        all_players.append(player_id)
    return {
        "radiant_players": radiant_players,
        "dire_players": dire_players,
        "all_players": all_players
    }


def get_match_stats(match_id) -> list:
    match_data = data_analysis.MATCHES[match_id]
    player_stats = []
    for player in match_data["players"]:
        player_stat = {}
        for stat in Player.Stats:
            try:
                player_stat[stat] = player[stat.value]
            except KeyError:
                player_stat[stat] = None
        player_stat["slot"] = get_player_slot_as_int(player["player_slot"])
        player_stat["id"] = player["account_id"]
        player_stat["hero_id"] = player["hero_id"]
        player_stat["items"] = [player["item_0"], player["item_1"], player["item_2"],
                                player["item_3"], player["item_4"], player["item_5"]]
        player_stat["data"] = player
        player_stats.append(player_stat)
    return player_stats


def update_stats(match_id):
    match_data = get_match_data(match_id)
    player_ids = get_player_ids(match_id)

    # winrate rating
    radiant_ratings = []
    dire_ratings = []
    players = {}
    for player_id in player_ids["all_players"]:
        players[player_id] = {}
        players[player_id]["stats"] = {}
        players[player_id]["towers"] = {}
        players[player_id]["barracks"] = {}

    for player_id in player_ids["radiant_players"]:
        radiant_ratings.append(Player.get_player(player_id).winrate)
    for player_id in player_ids["dire_players"]:
        dire_ratings.append(Player.get_player(player_id).winrate)
    if match_data["radiant_win"]:
        ranks = [0, 1]
    else:
        ranks = [1, 0]
    new_radiant_ratings, new_dire_ratings = rate([tuple(radiant_ratings), tuple(dire_ratings)], ranks=ranks)
    for idx, player_id in enumerate(player_ids["radiant_players"]):
        players[player_id]["winrate"] = new_radiant_ratings[idx]
    for idx, player_id in enumerate(player_ids["dire_players"]):
        players[player_id]["winrate"] = new_dire_ratings[idx]

    # other stats
    stats = get_match_stats(match_id)
    for stat in Player.Stats:
        # old match data doesn't have all required stats, so we skip them
        if not stats[0][stat]:
            for player_id in players:
                players[player_id]["stats"][stat] = Player.get_player(player_id).stats[stat]
            continue
        # since we are ranking, we want to have them all except deaths in descending order
        if stat == Player.Stats.DEATHS:
            reverse = False
        else:
            reverse = True
        stats.sort(key=lambda x: x[stat], reverse=reverse)
        ranks = []
        # consider each player individually and get their ranks in each stat/category
        for i in range(0, 10):
            ranks.append(next(k for k, v in enumerate(stats) if v["slot"] == i))

        ratings = []
        stats.sort(key=lambda x: x["slot"])
        for stat_2 in stats:
            ratings.append((Player.get_player(stat_2["id"]).stats[stat],))

        results = rate(ratings, ranks=ranks)
        # since new ratings are returned, we need to replace old ones
        for idx2, rating in enumerate(results):
            players[stats[idx2]["id"]]["stats"][stat] = rating[0]

    for player_stat in stats:
        players[player_stat["id"]]["hero_id"] = player_stat["hero_id"]
        players[player_stat["id"]]["items"] = player_stat["items"]
        players[player_stat["id"]]["data"] = player_stat["data"]

    # towers
    radiant_towers = {}
    dire_towers = {}
    for tower in Player.Towers:
        if tower.value & match_data["tower_status_radiant"]:
            radiant_towers[tower] = 1
        else:
            radiant_towers[tower] = 0
        if tower.value & match_data["tower_status_dire"]:
            dire_towers[tower] = 1
        else:
            dire_towers[tower] = 0

    # barracks
    radiant_barracks = {}
    dire_barracks = {}
    for barrack in Player.Barracks:
        if barrack.value & match_data["barracks_status_radiant"]:
            radiant_barracks[barrack] = 1
        else:
            radiant_barracks[barrack] = 0
        if barrack.value & match_data["barracks_status_dire"]:
            dire_barracks[barrack] = 1
        else:
            dire_barracks[barrack] = 0

    # finally update all stats
    for player_id in players:
        if player_id in player_ids["radiant_players"]:
            towers = radiant_towers
            barracks = radiant_barracks
        else:
            towers = dire_towers
            barracks = dire_barracks
        player = players[player_id]
        Player.get_player(player_id).update(player["winrate"], match_id, match_data["duration"], player["stats"],
                                            towers, barracks, player["hero_id"], player["items"], player["data"])


# Take from the first issue on the github of trueskill
def win_probability(a, b):
    deltaMu = sum([x.mu for x in a]) - sum([x.mu for x in b])
    sumSigma = sum([x.sigma ** 2 for x in a]) + sum([x.sigma ** 2 for x in b])
    playerCount = len(a) + len(b)
    denominator = math.sqrt(playerCount * (trueskill.BETA * trueskill.BETA) + sumSigma)
    return cdf(deltaMu / denominator)


def predict_outcome(match_id):
    """

    :param match_id:
    :type match_id: int
    :return: Probability that radiant will win
    :rtype: float
    """
    match_data = data_analysis.MATCHES[match_id]
    radiant_team = []
    dire_team = []
    for player in match_data["players"]:
        rating = Player.get_player(player["account_id"]).winrate
        if player["player_slot"] < 5:
            radiant_team.append(rating)
        else:
            dire_team.append(rating)
    return win_probability(radiant_team, dire_team)


def get_player_slot_as_int(player_slot):
    if player_slot < 128:
        return player_slot
    else:
        return player_slot - 123  # - 128 + 5


def get_player_side(player_slot):
    """
    :param player_slot:
    :type player_slot: int
    :return: Returns true if radiant
    :rtype: bool
    """
    mask = 0b10000000
    return mask & player_slot == 0


def radiant_win(match_id):
    return data_analysis.MATCHES[match_id]["radiant_win"]
