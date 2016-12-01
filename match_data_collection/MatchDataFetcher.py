import dota2api
import json
import time
import os.path

api = dota2api.Initialise("1DDE9137861534F5BB59A12E9CAA8511")
match_list = open("sorted_match_ids.txt", 'r').readlines()
counter = 0
for match in match_list:
    counter += 1
    match_id = int(match.rstrip('\n'))
    if counter % 100 == 0:
        print(float(counter) / match_list.__len__())
    if os.path.isfile('all_matches/' + str(match_id) + '.json'):
        continue
    time.sleep(0.1)
    file = open('all_matches/' + str(match_id) + '.json', 'w')
    match_data = api.get_match_details(match_id=match_id)
    json.dump(match_data, file)
