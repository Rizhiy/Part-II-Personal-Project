import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge

import data_analysis.Learning
from data_analysis import Learning
from data_analysis.Player import Stats

USE_SAVED_DATA = True
dataset_name = 'dataset.pkl'

if not (os.path.isfile(dataset_name) and USE_SAVED_DATA):
    Learning.generate_features(data_analysis.MATCH_LIST, dataset_name)
dataset = pickle.load(open(dataset_name, 'rb'))

regr = Ridge()
for stat in Stats:
    results = Learning.test_stat(stat, regr, dataset)

    print()
    print("Stat: {}".format(stat.value))
    print("{:>7} = {}\n{:>7} = {}\n{:>7} = {}".format("Mean", np.mean(results),
                                                      "Highest", max(results),
                                                      "Lowest", min(results)))
