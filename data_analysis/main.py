import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from data_analysis import Player, Match
import data_analysis
import json
import pickle

# temp fix, not fixing since not gonna use this in the future
import warnings

USE_SAVED_DATA = False

warnings.filterwarnings("ignore", category=DeprecationWarning)

MATCH_LIST = json.loads(open("../match_stats.json", 'r').read())["matches500"]

classifier_name = 'classifier.pkl'
dataset_name = 'dataset.pkl'

if os.path.isfile(dataset_name) and USE_SAVED_DATA:
    dataset = pickle.load(open(dataset_name, 'rb'))
else:
    X = []
    y = []
    counter = 0
    for match_id in MATCH_LIST:
        counter += 1
        if (counter % 10 == 0):
            print(float(counter) / len(MATCH_LIST))
        results = Match.generate_feature_set(match_id)
        X.append(results["features"])
        y.append(results["result"])

    dataset = {'X': X, 'y': y}

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)

X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['y'], test_size=0.1)
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100, 100))
clf.fit(X_train, y_train)

MATCH_LIST = json.loads(open("../match_stats.json", 'r').read())["matches500"]

print(clf.score(X_test, y_test)
