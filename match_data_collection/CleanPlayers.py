f = open('matches_ids.txt','r')

player_set = set()
players = f.read().splitlines()
for player in players:
    player_set.add(int(player))

player_list = list(player_set)
player_list.sort()
print(player_list)

clean_players = open('sorted_match_ids.txt','w')

for player in player_list:
    print(player,file=clean_players)