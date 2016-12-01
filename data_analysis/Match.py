import json
from data_analysis import Player

def generate_feature_set(match_id):
    match_data = get_match_data(match_id)
    radiant_players = []
    dire_players = []
    for player in match_data["players"]:
        if(Player.get_player_side(player["player_slot"])):
            radiant_players.append(player["account_id"])
        else:
            dire_players.append(player["account_id"])
    players = radiant_players + dire_players
    features = []
    for player in players:
        features += Player.Player(player,match_id-1).get_features()
    return {
        "features": features,
        "result": match_data["radiant_win"]
    }


def get_match_data(match_id):
    return json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))