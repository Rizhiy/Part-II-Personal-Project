from enum import Enum

import numpy
from trueskill import Rating

import data_analysis
from data_analysis import Match
from data_analysis import StatRatings


class Stats(Enum):
    KILLS = "kills"
    DEATHS = "deaths"
    ASSISTS = "assists"
    LEVEL = "level"
    GPM = "gold_per_min"
    XPM = "xp_per_min"
    CREEPS = "last_hits"
    DENIES = "denies"
    TOWER_DMG = "tower_damage"
    HERO_DMG = "hero_damage"
    HEALING = "hero_healing"


class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.winrate = Rating()
        self.stats = {}
        self.total_games = 0
        self.last_match_processed = 0
        for stat in Stats:
            self.stats[stat] = StatRatings.StatRatings()

    def __str__(self):
        result = self.short_string()
        stats = []
        for key, value in self.stats.items():
            stats.append("{:>12} : {}".format(key.value, str(value)))
        result += "\n".join(stats)
        return result

    def get_all_matches(self):
        match_list = []
        for match_id in data_analysis.MATCH_LIST:
            for player in Match.get_match_data(match_id)["players"]:
                if self.player_id == player['account_id']:
                    match_list.append(match_id)
        return match_list

    def short_string(self):
        return "{:>12} = {}\n{:>12} = {}\n{:>12} = {}\n".format("account id", self.player_id, "winrate", self.winrate,
                                                                "total games", self.total_games)


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
        if total_games > 5:
            break
        match_data = Match.get_match_data(match)
        for player in match_data["players"]:
            if (player["account_id"] == player_id):
                if match_data["radiant_win"] == Match.get_player_side(player["player_slot"]):
                    wins += 1
                assists.append(player["assists"])
                kills.append(player["kills"])
                level.append(player["level"])
                gpm.append(player["gold_per_min"])
                denies.append(player["denies"])
                deaths.append(player["deaths"])
                xpm.append(player["xp_per_min"])
                total_games += 1
    if (total_games):
        return {
            "winrate": wins / total_games,
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


def get_player(player_id):
    if not player_id in data_analysis.PLAYERS:
        data_analysis.PLAYERS[player_id] = Player(player_id)
    return data_analysis.PLAYERS[player_id]
