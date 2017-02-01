from enum import Enum

import matplotlib.pyplot as plt
import data_analysis
import numpy as np
from scipy.stats import gaussian_kde

from data_analysis import Learning


def raw_stat_hist(stat: Enum):
    data = []
    for match in data_analysis.MATCH_LIST:
        for player in data_analysis.MATCHES[match]["players"]:
            data.append(player[stat.value])
    bins = max(data)
    plt.figure()
    plt.title(stat.value)
    if bins < 50:
        plt.hist(data, bins=bins, normed=True)
    else:
        density = gaussian_kde(data)
        xs = np.linspace(0, bins, 128)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        plt.plot(xs, density(xs))
    plt.draw()


def error_stat_hist(stat: Enum, regr, dataset):
    results = np.array(Learning.predict_stat(stat, regr, dataset))
    results = results[~np.isnan(results)] # Remove NaN
    density = gaussian_kde(results)
    xs = np.linspace(min(results), max(results), 128)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    raw_stat_hist(stat)
    stat_error, = plt.plot(xs, density(xs),label="Error distribution + mean")
    plt.legend(handles=[stat_error])
    plt.draw()
