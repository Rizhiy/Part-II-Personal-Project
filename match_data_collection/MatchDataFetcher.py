import json
import os.path
import sys
import time

import dota2api

api = dota2api.Initialise("1DDE9137861534F5BB59A12E9CAA8511")
match_list = open("sorted_match_ids.txt", 'r').readlines()
counter = 0
for match in match_list:
    print("\rProgress: {:0.3f}".format(float(counter) / match_list.__len__()), end="")
    counter += 1
    match_id = int(match.rstrip('\n'))
    if match_id > 2837037509:
        continue
    if os.path.isfile('../all_matches/' + str(match_id) + '.json'):
        continue
    time.sleep(0.1)
    file = open('../all_matches/' + str(match_id) + '.json', 'w')
    match_data = api.get_match_details(match_id=match_id)
    json.dump(match_data, file)