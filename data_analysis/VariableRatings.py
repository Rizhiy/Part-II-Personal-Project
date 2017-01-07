import data_analysis
from trueskill import Rating


class VariableRatings:
    own = Rating()
    highest_team = Rating()
    median_team = Rating()
    lowest_team = Rating()
    highest_enemy = Rating()
    median_enemy = Rating()
    lowest_enemy = Rating()
