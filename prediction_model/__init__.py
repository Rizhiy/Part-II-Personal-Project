import json
import pprint

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

PLAYERS = {}

PP = pprint.PrettyPrinter(indent=2)

PLAYER_DIM = 10
TEAM_DIM = 10
PLAYERS_PER_TEAM = 5
NUM_OF_TEAMS = 2
PLAYER_RESULT_DIM = 8
TEAM_RESULTS_DIM = 2
BATCH_SIZE = 1

print("Initialisation finished")
