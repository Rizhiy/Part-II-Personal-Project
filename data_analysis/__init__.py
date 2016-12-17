import json
from pymongo import MongoClient

GAMES_TO_USE = "matches200"
MATCH_FOLDER = "../all_matches"

MATCH_LIST = json.loads(open("../match_stats.json",'r').read())[GAMES_TO_USE]
MATCH_LIST.sort()

DENDI_ID = 70388657
FEAR_ID = 87177591
PUPPEY_ID = 87278757
S4_ID = 41231571
EE_ID = 43276219
TI_6_LAST_GAME_ID = 2569610900

DB = MongoClient().PersonalProject

MATCHES = {}
for match_id in MATCH_LIST:
    MATCHES[match_id] = json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))