import data_analysis
from enum import Enum
from trueskill import Rating


class RatingType(Enum):
    HIGHEST = "Highest"
    MEDIAN = "Median"
    LOWEST = "Lowest"
    TOTAL = "Total"

class StatRatings:
    """
    Own stats are considered out of all 10 players.
    Team stats are team based, i.e whether highest/median/lowest rating of allies is higher than enemies.
    Total is not based on rating, but actual value.
    """

    def __init__(self):
        self.own = Rating()
        self.team_ratings = {}
        for type in RatingType:
            self.team_ratings[type] = Rating()

    def __str__(self):
        team_ratings = ""
        for rating_type in RatingType:
            team_ratings += "{:>7} = {}".format(rating_type.value,self.team_ratings[rating_type])
        return "own = {}, {}".format(self.own, team_ratings)
