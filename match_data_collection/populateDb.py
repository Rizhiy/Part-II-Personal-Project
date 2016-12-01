import json

from pymongo import MongoClient
client = MongoClient()

db = client.PersonalProject

MATCH_LIST = json.loads(open("../match_stats.json",'r').read())["allMatches"]


for index,match_id in enumerate(MATCH_LIST):
    if(match_id % 100 == 0):
        print(index/len(MATCH_LIST))
    match_data = json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))
    match_data["_id"] = match_id
    db.matches.insert_one(match_data)