import os
import sys
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from data_analysis import Match
import data_analysis
import pickle

# temp fix, not fixing since not gonna use this in the future
import warnings

USE_SAVED_DATA = False

warnings.filterwarnings("ignore", category=DeprecationWarning)

classifier_name = 'classifier.pkl'
dataset_name = 'dataset.pkl'

if os.path.isfile(dataset_name) and USE_SAVED_DATA:
    dataset = pickle.load(open(dataset_name, 'rb'))
else:
    X = []
    y = []
    counter = 0
    for match_id in data_analysis.MATCH_LIST:
        counter += 1
        if (counter % 10 == 0):
            print("\rPreparing dataset: " + str(float(counter) / len(data_analysis.MATCH_LIST)), end='')
            sys.stdout.flush()
        results = Match.generate_feature_set(match_id)
        X.append(results["features"])
        y.append(results["result"])
    print("\rPreparing dataset: Done")
    # X = np.array(X)
    # X = preprocessing.scale(X)
    dataset = {'X': X, 'y': y}

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
clf = MLPClassifier(solver='lbfgs',
                    hidden_layer_sizes=(100, 100, 100,))
print(np.mean(cross_val_score(clf, dataset['X'], dataset['y'], cv=10)))
