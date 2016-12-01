import json
import os.path

match_list = open("sorted_match_ids.txt", 'r').readlines()
allMatches = []
player_stats = {}
for match in match_list:
    match_id = int(match.rstrip('\n'))
    allMatches.append(match_id)
    if os.path.isfile('../all_matches/' + str(match_id) + '.json'):
        match_data = json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))
        for player in match_data["players"]:
            player_id = int(player["account_id"])
            if player_id in player_stats:
                player_stats[player_id] += 1
            else:
                player_stats[player_id] = 1

# don't judge
players20 = []
players50 = []
players100 = []
players200 = []
players500 = []
players1000 = []

for player_id in player_stats:
    if player_stats[player_id] > 1000:
        players1000.append(player_id)
    if player_stats[player_id] > 500:
        players500.append(player_id)
    if player_stats[player_id] > 200:
        players200.append(player_id)
    if player_stats[player_id] > 100:
        players100.append(player_id)
    if player_stats[player_id] > 50:
        players50.append(player_id)
    if player_stats[player_id] > 20:
        players20.append(player_id)

matches20 = []
matches50 = []
matches100 = []
matches200 = []
matches500 = []
matches1000 = []
for match in match_list:
    match_id = int(match.rstrip('\n'))
    if os.path.isfile('../all_matches/' + str(match_id) + '.json'):
        match_data = json.load(open('../all_matches/' + str(match_id) + '.json', 'r'))
        goodPlayers1000 = True
        goodPlayers500 = True
        goodPlayers200 = True
        goodPlayers100 = True
        goodPlayers50 = True
        goodPlayers20 = True
        if(len(match_data["players"]) != 10):
            continue
        for player in match_data["players"]:
            player_id = int(player["account_id"])
            if player_id not in players1000:
                goodPlayers1000 = False
            if player_id not in players500:
                goodPlayers500 = False
            if player_id not in players200:
                goodPlayers200 = False
            if player_id not in players100:
                goodPlayers100 = False
            if player_id not in players50:
                goodPlayers50 = False
            if player_id not in players20:
                goodPlayers20 = False
        if goodPlayers1000:
            matches1000.append(match_id)
        if goodPlayers500:
            matches500.append(match_id)
        if goodPlayers200:
            matches200.append(match_id)
        if goodPlayers100:
            matches100.append(match_id)
        if goodPlayers50:
            matches50.append(match_id)
        if goodPlayers20:
            matches20.append(match_id)

match_stats = {
    "allMatches": allMatches,
    "matches1000": matches1000,
    "matches500": matches500,
    "matches200": matches200,
    "matches100": matches100,
    "matches50": matches50,
    "matches20": matches20
}

print(matches20.__len__())
print(matches50.__len__())
print(matches100.__len__())
print(matches200.__len__())
print(matches500.__len__())
print(matches1000.__len__())

file = open("../match_stats.json",'w')
json.dump(match_stats,file)
