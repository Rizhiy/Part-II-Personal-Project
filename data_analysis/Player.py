import json
import numpy
import data_analysis

from data_analysis import Match

class Player:
    def __str__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join('%s=%s' % item for item in vars(self).items()))

    def __init__(self, player_id, game_number=0):
        self.player_id = player_id
        self.winrate = 0
        self.assists = 0
        self.kills = 0
        self.level = 0
        self.gpm = 0
        self.denies = 0
        self.deaths = 0
        self.xpm = 0
        self.total_games = 0
        self.last_match_processed = 0
        self.update_player(game_number)

    def update_player(self, game_number):
        results = calculate_player_averages(self.player_id, data_analysis.MATCH_LIST, game_number,
                                            self.last_match_processed)
        if(results["total_games"]):
            self.winrate = weighted_average(self.winrate,self.total_games, results["winrate"],results["total_games"])
            self.assists = weighted_average(self.assists, self.total_games, results["assists"], results["total_games"])
            self.kills = weighted_average(self.kills, self.total_games, results["kills"], results["total_games"])
            self.level = weighted_average(self.level, self.total_games, results["level"], results["total_games"])
            self.gpm = weighted_average(self.gpm, self.total_games, results["gpm"], results["total_games"])
            self.denies = weighted_average(self.denies, self.total_games, results["denies"], results["total_games"])
            self.deaths = weighted_average(self.deaths, self.total_games, results["deaths"], results["total_games"])
            self.xpm = weighted_average(self.xpm, self.total_games, results["xpm"], results["total_games"])
            self.total_games += results["total_games"]
            self.last_match_processed = game_number

    def get_features(self):
        return [self.winrate,self.assists,self.kills,self.level,self.gpm,self.deaths,self.denies,self.xpm,self.total_games]


def calculate_player_averages(player_id, matches_to_use, max_match_id, min_match_id=0):
    wins = 0
    assists = []
    kills = []
    level = []
    gpm = []
    denies = []
    deaths = []
    xpm = []
    total_games = 0
    matches_to_use.sort(reverse=True)
    for match in matches_to_use:
        if match > max_match_id:
            continue
        if total_games > 40:
            break
        match_data = Match.get_match_data(match)
        for player in match_data["players"]:
            if (player["account_id"] == player_id):
                if match_data["radiant_win"] == get_player_side(player["player_slot"]):
                    wins += 1
                assists.append(player["assists"])
                kills.append(player["kills"])
                level.append(player["level"])
                gpm.append(player["gold_per_min"])
                denies.append(player["denies"])
                deaths.append(player["deaths"])
                xpm.append(player["xp_per_min"])
                total_games += 1
    if(total_games):
        return {
            "winrate": wins/total_games,
            "assists": numpy.mean(assists),
            "kills": numpy.mean(kills),
            "level": numpy.mean(level),
            "gpm": numpy.mean(gpm),
            "denies": numpy.mean(denies),
            "deaths": numpy.mean(deaths),
            "xpm": numpy.mean(xpm),
            "total_games": total_games
        }
    else:
        return {
            "winrate": 0,
            "assists": 0,
            "kills": 0,
            "level": 0,
            "gpm": 0,
            "denies": 0,
            "deaths": 0,
            "xpm": 0,
            "total_games": 0
        }


def weighted_average(value_one, weight_one, value_two, weight_two):
    return (value_one * weight_one + value_two * weight_two) / (weight_one + weight_two)

# returns true if radiant
def get_player_side(player_slot):
    mask = 0b10000000
    return mask & player_slot == 0