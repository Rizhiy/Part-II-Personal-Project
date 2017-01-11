import data_analysis
from enum import Enum
from trueskill import Rating

class StatRatings:
    """
    Own stats are considered out of all 10 players.
    Team stats are team based, i.e whether highest/median/lowest rating of allies is higher than enemies.
    Total is not based on rating, but actual value.
    """

    def __init__(self):
        self.own = Rating()

    def __str__(self):
        return "own = {}".format(self.own)
