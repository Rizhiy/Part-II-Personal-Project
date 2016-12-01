import os

from sklearn.neural_network import MLPClassifier
from data_analysis import Player, Match
import data_analysis
import json
import pickle

# temp fix, not fixing since not gonna use this in the future
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

MATCH_LIST = json.loads(open("../match_stats.json",'r').read())["matches1000"]

classifier_name = 'classifier.pkl'

if not os.path.isfile(classifier_name):
    X = []
    Y = []
    counter = 0
    for match_id in MATCH_LIST:
        counter += 1
        if(counter % 10 == 0):
            print(float(counter)/len(MATCH_LIST))
        results = Match.generate_feature_set(match_id)
        X.append(results["features"])
        Y.append(results["result"])



    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X,Y)

    with open(classifier_name, 'wb') as output:
        pickle.dump(clf,output,pickle.HIGHEST_PROTOCOL)


clf = pickle.load(open(classifier_name, 'rb'))

MATCH_LIST = json.loads(open("../match_stats.json",'r').read())["matches500"]

correct = 0
total = 0
for idx,match_id in enumerate(MATCH_LIST):
    if(idx % 100 == 0):
        print(float(idx)/len(MATCH_LIST))
    if(idx > 1000):
        break
    features = Match.generate_feature_set(match_id)
    total += 1
    if clf.predict(features["features"]).all() == features["result"]:
        correct += 1

print(float(correct)/total)