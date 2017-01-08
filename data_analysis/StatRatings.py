import data_analysis
from trueskill import Rating


class StatRatings:
    """
    Own stats are considered out of all 10 players
    Team stats are team based, i.e whether highest/median/lowest rating of allies is higher than enemies
    """
    def __init__(self):
        self.own = Rating()
        self.highest_team = Rating()
        self.median_team = Rating()
        self.lowest_team = Rating()
        self.highest_enemy = Rating()
        self.median_enemy = Rating()
        self.lowest_enemy = Rating()
