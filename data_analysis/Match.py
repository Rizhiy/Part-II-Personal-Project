import random

from data_analysis import Player

import data_analysis
from trueskill import rate


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
        player_objects.append(Player.Player(player))
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
    :rtype: dict
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


def get_match_stats(match_id):
    match_data = data_analysis.MATCHES[match_id]
    player_stats = []
    for player in match_data["players"]:
        player_stat = {}
        for stat in Player.Stats:
            try:
                player_stat[stat] = player[stat.value]
            except KeyError:
                player_stat[stat] = None
        player_stat["slot"] = Player.get_player_slot_as_int(player["player_slot"])
        player_stat["id"] = player["account_id"]
        player_stats.append(player_stat)
    return player_stats


def update_stats(match_id):
    player_ids = get_player_ids(match_id)
    radiant_ratings = []
    dire_ratings = []
    for player_id in player_ids["all_players"]:
        if not player_id in data_analysis.PLAYERS:
            data_analysis.PLAYERS[player_id] = Player.Player(player_id)
    for player_id in player_ids["radiant_players"]:
        radiant_ratings.append(data_analysis.PLAYERS[player_id].winrate)
    for player_id in player_ids["dire_players"]:
        dire_ratings.append(data_analysis.PLAYERS[player_id].winrate)

    # winrate rating
    if get_match_data(match_id)["radiant_win"]:
        ranks = [0, 1]
    else:
        ranks = [1, 0]
    new_radiant_ratings, new_dire_ratings = rate([tuple(radiant_ratings), tuple(dire_ratings)], ranks=ranks)
    for idx, player_id in enumerate(player_ids["radiant_players"]):
        data_analysis.PLAYERS[player_id].winrate = new_radiant_ratings[idx]
    for idx, player_id in enumerate(player_ids["dire_players"]):
        data_analysis.PLAYERS[player_id].winrate = new_radiant_ratings[idx]

    # other stats
    stats = get_match_stats(match_id)
    for stat in Player.Stats:
        # old match data doesn't have all required stats, so we skip them
        if not stats[0][stat]:
            continue
        # since we are ranking, we want to have them all except deaths in descending order
        if stat == Player.Stats.DEATHS:
            reverse = False
        else:
            reverse = True
        stats.sort(key=lambda x: x[stat], reverse=reverse)
        ranks = []
        # consider each player individually and get their ranks in each stat/category
        for j in range(0, 10):
            ranks.append(next(i for i, v in enumerate(stats) if v["slot"] == j))

        ratings = []
        stats.sort(key=lambda x: x["slot"])
        for stat_2 in stats:
            ratings.append((data_analysis.PLAYERS[stat_2["id"]].stats[stat].own,))
        results = rate(ratings, ranks=ranks)
        # since new ratings are returned, we need to replace old ones
        for idx2, rating in enumerate(results):
            data_analysis.PLAYERS[stats[idx2]["id"]].stats[stat].own = rating[0]

    for player_id in player_ids["all_players"]:
        player = data_analysis.PLAYERS[player_id]
        player.total_games += 1
        player.last_match_processed = match_id