import json
import pprint

import numpy as np
import tensorflow as tf

GAMES_TO_USE = "matches50"
MATCH_FOLDER = "../all_matches"

MATCH_LIST = json.loads(open("../match_stats.json", 'r').read())[GAMES_TO_USE]
MATCH_LIST.sort()

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

# Since we don't have a lot of games, we can just load all of them into memory
MATCHES = {}
for match_id in MATCH_LIST:
    MATCHES[match_id] = json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))

PLAYER_SKILLS = {}
PLAYER_PERFORMANCES = {}

PP = pprint.PrettyPrinter(indent=2)

BATCH_SIZE = 1
PLAYER_RESULT_DIM = 8
TEAM_RESULTS_DIM = 2
PLAYER_DIM = int(.5 * PLAYER_RESULT_DIM)
TEAM_DIM = 2 * PLAYER_DIM

PLAYERS_PER_TEAM = 5
NUM_OF_TEAMS = 2

# Print options for numpy arrays
np.set_printoptions(precision=3, suppress=True)

# Make an array with games for each player
PLAYER_GAMES = {}
for match_id in MATCH_LIST:
    for player in MATCHES[match_id]["players"]:
        player_id = player["account_id"]
        if player_id not in PLAYER_GAMES:
            PLAYER_GAMES[player_id] = {}
            PLAYER_GAMES[player_id]["all"] = []
        player_set = PLAYER_GAMES[player_id]
        if len(player_set["all"]) > 0:
            prev_match_id = player_set["all"][-1]
            player_set[prev_match_id]["next"] = match_id
            player_set[match_id] = {"prev": prev_match_id}
        else:
            player_set[match_id] = {"prev": None}
        player_set[match_id]["slot"] = player["player_slot"]
        player_set["all"].append(match_id)

# Initialise TensorFlow session
SESSION = tf.Session()

print("Initialisation finished")
