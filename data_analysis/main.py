import os
import sys
import numpy as np
import data_analysis
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from data_analysis import Match, Player

from trueskill import Rating
from trueskill import expose

# temp fix, not fixing since not gonna use this in the future
import warnings

USE_SAVED_DATA = True

warnings.filterwarnings("ignore", category=DeprecationWarning)

dataset_name = 'dataset.pkl'

if os.path.isfile(dataset_name) and USE_SAVED_DATA:
    dataset = pickle.load(open(dataset_name, 'rb'))
else:
    X = []
    y = []
    match_list = data_analysis.MATCH_LIST
    match_list.sort()
    for counter, match_id in enumerate(match_list):
        print("\rPreparing dataset: " + ("{:.2f}".format((float(counter) / len(match_list)))), end='')
        sys.stdout.flush()
        results = Match.generate_feature_set(match_id)
        X.append(results["features"])
        y.append(results["target"])
        Match.update_stats(match_id)
    print("\rPreparing dataset: Done")

    X = np.array(X)
    dataset = {'X': X, 'y': y}

    with open(dataset_name, 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)

regr = Ridge()
print(np.mean(cross_val_score(regr, dataset['X'], dataset['y'], cv=10)))
