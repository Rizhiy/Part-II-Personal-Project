import json
import pprint
from enum import Enum

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.core import K


class Stats(Enum):
    KILLS = "kills"
    DEATHS = "deaths"
    ASSISTS = "assists"
    LEVEL = "level"
    GPM = "gold_per_min"
    XPM = "xp_per_min"
    CREEPS = "last_hits"
    DENIES = "denies"
    # Those three values are missing from first half of the dataset,
    # TOWER_DMG = "tower_damage"
    # HERO_DMG = "hero_damage"
    # HEALING = "hero_healing"


# ids of some experienced players
DENDI_ID = 70388657
FEAR_ID = 87177591
PUPPEY_ID = 87278757
S4_ID = 41231571
RESOLUTION_ID = 86725175
EE_ID = 43276219
MISERY_ID = 87382579
MIRACLE_ID = 105248644
# id of a reference game
TI_6_LAST_GAME_ID = 2569610900
TI_5_LAST_GAME_ID = 1697818230
MANILA_MAJOR_LAST_GAME_ID = 2430984806
BOSTON_MAJOR_LAST_GAME_ID = 2837037509
TEST_TEAM = [{"p_id": DENDI_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": DENDI_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": PUPPEY_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": S4_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": EE_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": MISERY_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": MIRACLE_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": RESOLUTION_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": FEAR_ID, "m_id": TI_6_LAST_GAME_ID},
             {"p_id": S4_ID, "m_id": TI_6_LAST_GAME_ID}, ]

# Settings for the model
BATCH_SIZE = 128

PLAYER_RESULT_DIM = 8
TEAM_RESULTS_DIM = 2

GAMES_TO_CONSIDER = 1
PLAYER_DIM = 4
TEAM_DIM = 8

DEBUG = 0

RETRAIN = True

PLAYERS_PER_TEAM = 5
NUM_OF_TEAMS = 2

DEFAULT_PLAYER_SKILL = [0] * PLAYER_DIM + [1] * PLAYER_DIM

GAMES_TO_USE = "matches50"
MATCH_FOLDER = "../all_matches"

MATCH_LIST = json.loads(open("../match_stats.json", 'r').read())[GAMES_TO_USE]
NUMBER_OF_BATCHES = int(len(MATCH_LIST) / BATCH_SIZE)

# Since we don't have a lot of games, we can just load all of them into memory
MATCHES = {}
for match_id in MATCH_LIST:
    MATCHES[match_id] = json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))

PLAYER_SKILLS = {}
PLAYER_SKILLS_MU = {}
PLAYER_PERFORMANCES = {}

PP = pprint.PrettyPrinter(indent=2)

# Print options for numpy arrays
np.set_printoptions(precision=3, suppress=True)

# Make an array with games for each player
PLAYER_GAMES = {}

# Initialise TensorFlow session
SESSION = tf.Session()

K.set_learning_phase(True)

LOSS_MULTIPLIER = [1] * PLAYER_RESULT_DIM
# LOSS_MULTIPLIER[0] = 10

VARIABLE_ORDER = [Stats.GPM, Stats.XPM, Stats.CREEPS, Stats.DENIES, Stats.KILLS, Stats.DEATHS, Stats.ASSISTS,
                  Stats.LEVEL]

# Standardiser settings
SCALER = {}

# pyplot settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)

print("Initialisation finished")
