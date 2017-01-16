from enum import Enum

import numpy
from trueskill import Rating

import data_analysis
from data_analysis import Match


class Stats(Enum):
    KILLS = "kills"
    DEATHS = "deaths"
    ASSISTS = "assists"
    KDA = "KDA"
    LEVEL = "level"
    GPM = "gold_per_min"
    XPM = "xp_per_min"
    CREEPS = "last_hits"
    DENIES = "denies"
    # Those three values are missing from first half of the dataset,
    # TOWER_DMG = "tower_damage"
    # HERO_DMG = "hero_damage"
    # HEALING = "hero_healing"


class Towers(Enum):
    ANCIENT_BOTTOM = 0b0000010000000000
    ANCIENT_TOP = 0b0000001000000000
    BOTTOM_T3 = 0b0000000100000000
    BOOTOM_T2 = 0b0000000010000000
    BOTTOM_T1 = 0b0000000001000000
    MIDDLE_T3 = 0b0000000000100000
    MIDDLE_T2 = 0b0000000000010000
    MIDDLE_T1 = 0b0000000000001000
    TOP_T3 = 0b0000000000000100
    TOP_T2 = 0b0000000000000010
    TOP_T1 = 0b0000000000000001


class Barracks(Enum):
    BOTTOM_RANGED = 0b00100000
    BOTTOM_MELEE = 0b00010000
    MIDDLE_RANGED = 0b00001000
    MIDDLE_MELEE = 0b00000100
    TOP_RANGED = 0b00000010
    TOP_MELEE = 0b00000001


class Player:
    alpha = 0.9

    def __init__(self, player_id):
        self._player_id = player_id
        self.winrate = Rating()
        self.total_games = 0
        self.last_match_processed = 0
        self.game_length = 0
        self.stats = {}
        for stat in Stats:
            self.stats[stat] = Rating()
        self.towers = {}
        for tower in Towers:
            self.towers[tower] = 0
        self.barracks = {}
        for barrack in Barracks:
            self.barracks[barrack] = 0
        self.heroes = {}
        for i in range(1, 114):  # Those ranges are hard coded, need to fetch from db in future
            self.heroes[i] = 0
        self.items = {}
        for i in range(0, 266):
            self.items[i] = 0

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
                if self._player_id == player['account_id']:
                    match_list.append(match_id)
        return match_list

    def get_features(self):
        features = []
        features += [self.winrate.mu, self.winrate.sigma, self.total_games, self.game_length]
        for stat in self.stats:
            stat2 = self.stats[stat]
            features += [stat2.mu, stat2.sigma]
        for key, tower in self.towers.items():
            features.append(tower)
        for key, barrack in self.barracks.items():
            features.append(barrack)
        for hero_id, hero_stat in self.heroes.items():
            features.append(hero_stat)
        for item_id, item_stat in self.heroes.items():
            features.append(item_stat)
        return features

    def get_player_id(self):
        return self._player_id

    def short_string(self):
        return "{:>12} = {}\n{:>12} = {}\n{:>12} = {}\n".format("account id", self._player_id,
                                                                "winrate", self.winrate,
                                                                "total games", self.total_games)

    def update(self, winrate, match_id, game_length, stats, towers, barracks, hero_id, items):
        self.winrate = winrate
        self.last_match_processed = match_id
        self.game_length = exp_average(self.game_length, game_length, Player.alpha)
        self.stats = stats

        for tower in Towers:
            self.towers[tower] = exp_average(self.towers[tower], towers[tower], Player.alpha)
        for barrack in Barracks:
            self.barracks[barrack] = exp_average(self.barracks[barrack], barracks[barrack], Player.alpha)

        for hero, hero_stat in self.heroes.items():
            self.heroes[hero] = hero_stat * Player.alpha
            if hero == hero_id:
                self.heroes[hero] += 1
        for item_id, item_stat in self.items.items():
            self.items[item_id] = item_stat * Player.alpha
            if item_id in items:
                self.items[item_id] += 1

        self.total_games += 1


def exp_average(old, new, alpha):
    return (old * alpha + new) / (1 + alpha)


def get_player(player_id) -> Player:
    if player_id not in data_analysis.PLAYERS:
        data_analysis.PLAYERS[player_id] = Player(player_id)
    return data_analysis.PLAYERS[player_id]


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
            if player["account_id"] == player_id:
                if match_data["radiant_win"] == Match.get_player_side(player["player_slot"]):
                    wins += 1
                assists.append(player["assists"])
                kills.append(player["kills"])
                level.append(player["level"])
                gpm.append(player["gold_6per_min"])
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
