from enum import Enum

import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from scipy.stats import gaussian_kde, hmean

import data_analysis

from data_analysis import Learning


def density_hist(data: list, label="", figure=True, title=""):
    if figure:
        plt.figure()
        plt.title(title)

    bins = max(data) - min(data)
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN
    if bins < 50:
        plt.hist(data, bins=bins, normed=True)
    else:
        try:
            density = gaussian_kde(data)
        except MemoryError:
            density = gaussian_kde(data[::100])
        xs = np.linspace(min(data), max(data), 128)
        density.covariance_factor = lambda: .25
        density._compute_covariance()
        plt.plot(xs, density(xs), label=label)
    plt.draw()


def raw_stat_hist(stat: Enum):
    data = []
    for match in data_analysis.MATCH_LIST:
        for player in data_analysis.MATCHES[match]["players"]:
            data.append(player[stat.value])
    density_hist(data,"actual values",title=stat.value)


# TODO: there is quite a bit repetition in these methods need to refactor.
def error_stat_hist(stat: Enum, regr, dataset):
    results = np.array(Learning.calculate_error(stat, regr, dataset))
    results = results[~np.isnan(results)]  # Remove NaN
    density = gaussian_kde(results)
    xs = np.linspace(min(results), max(results), 128)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.figure()
    plt.title(stat.value + " error")
    plt.plot(xs, density(xs))
    plt.draw()


def prediction_hist(stat: Enum, regr, dataset):
    results = np.array(Learning.predict_stat(stat, regr, dataset))
    results = results[~np.isnan(results)]  # Remove NaN
    density = gaussian_kde(results)
    xs = np.linspace(min(results), max(results), 128)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    raw_stat_hist(stat)
    plt.plot(xs, density(xs), label="predicted values")
    plt.legend()
    plt.draw()


def raw_trueskill_winrate(dataset):
    outcomes = []
    radiant_sums = []
    dire_sums = []
    for idx, match_id in enumerate(dataset["match_ids"]):
        radiant_sum = []
        dire_sum = []
        for idx2, feature in enumerate(dataset["features"][idx]):
            if idx2 % 22 != 0:
                continue
            if idx2 / 22 < 5:
                radiant_sum.append(feature)
            else:
                dire_sum.append(feature)
        outcomes.append(data_analysis.MATCHES[match_id]["radiant_win"])
        radiant_sums.append(np.sum(radiant_sum))
        dire_sums.append(np.sum(dire_sum))
    plt.figure()
    plt.title("winrate")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("winrate")
    for idx, outcome in enumerate(outcomes):
        if outcome:
            ax.plot(radiant_sums[idx], dire_sums[idx], marker='s', color="red")
        else:
            ax.plot(radiant_sums[idx], dire_sums[idx], marker='o', color="green")
    plt.xlabel("Radiant TrueSkill sum")
    plt.ylabel("Dire TrueSkill sum")
    # PdfPages('winrate.pdf').savefig()
    plt.draw()


def plot_estimation(mean,std,title,value):
    x = np.linspace(mean-std*4,mean+std*4,100)
    plt.figure()
    plt.title(title)
    plt.plot(x,mlab.normpdf(x,mean,std),label="Predicted probability:\n mu={}, sigma={}".format(int(mean),int(std)))
    plt.axvline(x=value,label="Actual value: {}".format(value),color='r')
    plt.ylim([0,0.004])
    plt.legend()
