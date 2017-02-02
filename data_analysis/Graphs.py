from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from scipy.stats import gaussian_kde, hmean

import data_analysis

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
    results = results[~np.isnan(results)]  # Remove NaN
    density = gaussian_kde(results)
    xs = np.linspace(min(results), max(results), 128)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.figure()
    plt.title(stat.value + " error")
    plt.plot(xs, density(xs))
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
