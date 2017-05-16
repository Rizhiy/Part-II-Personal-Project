from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab
from scipy.stats import gaussian_kde, gamma, norm, chisquare

import data_analysis
from data_analysis import Learning


def density_hist(data: list, label=""):
    bins = max(data) - min(data)
    if bins < 10:
        bins = 50
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN
    if bins < 20:
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


def fit_gaussian(stat: Enum):
    data = []
    for match in data_analysis.MATCH_LIST:
        for player in data_analysis.MATCHES[match]["players"]:
            data.append(player[stat.value])
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    variance = np.var(data)
    std = np.sqrt(variance)
    raw_stat_hist(stat)
    x = np.linspace(mean - std * 4, mean + std * 4, 128)
    plt.plot(x, mlab.normpdf(x, mean, std),
             label="Fitted gaussian:\n mu={}, sigma={}".format(int(mean), int(std)))
    plt.draw()


def fit_gamma_and_gaussian(stat: Enum):
    data = []
    for match in data_analysis.MATCH_LIST:
        for player in data_analysis.MATCHES[match]["players"]:
            data.append(player[stat.value])
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    variance = np.var(data)
    k = np.square(mean) / variance
    theta = variance / mean
    max_value = np.percentile(data, 99)
    bins = int(max_value / 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=16)
    x = np.linspace(min(data), max_value, bins)
    plt.hist(data, x, color='white')
    param = gamma.fit(data)
    y1 = gamma.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * len(data) * max_value / bins
    hist = np.histogram(data, x)
    plt.plot(x, y1, label=r"Fitted gamma:\\ $k = {:3.2f}$, $\theta = {:3.1f}$\\ $\chi^2 = {}$".
             format(k, theta, int(chisquare(hist[0], y1[:-1])[0])))
    param = norm.fit(data)
    y2 = norm.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * len(data) * max_value / bins
    plt.plot(x, y2, '--', label=r"Fitted gaussian:\\ $\mu = {}$, $\sigma = {}$\\ $\chi^2 = {}$".
             format(int(param[-2]), int(param[-1]), int(chisquare(hist[0], y2[:-1])[0])))
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.legend()
    plt.draw()


def raw_stat_hist(stat: Enum):
    data = []
    for match in data_analysis.MATCH_LIST:
        for player in data_analysis.MATCHES[match]["players"]:
            data.append(player[stat.value])
    max = np.percentile(data, 99)
    data = np.array(data)
    data = data[data < max]
    density_hist(data, "Distribution of values")
    # value = np.percentile(data, 99, 0)
    # plt.axvline(x=value, label="99% percentile", color='r', linestyle='--')
    # value = np.percentile(data, 1, 0)
    # plt.axvline(x=value, label="1% percentile", color='g', linestyle='-.')
    plt.legend()


# TODO: there is quite a bit repetition in these methods need to refactor.
def error_stat_hist(stat: Enum, regr, dataset):
    results = np.array(Learning.calculate_error(stat, regr, dataset))
    results = results[~np.isnan(results)]  # Remove NaN
    density = gaussian_kde(results)
    xs = np.linspace(min(results), max(results), 128)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
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
    plt.plot(xs, density(xs), label="Predicted values")
    plt.legend(prop={'size': 20})
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.draw()


def raw_trueskill_winrate(dataset):
    outcomes = []
    radiant_sums = []
    dire_sums = []
    for idx, match_id in enumerate(dataset["match_ids"]):
        if match_id not in data_analysis.MATCHES:
            continue
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
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_title("winrate")
    for idx, outcome in enumerate(outcomes):
        if outcome:
            ax.plot(radiant_sums[idx], dire_sums[idx], marker='s', color="red")
        else:
            ax.plot(radiant_sums[idx], dire_sums[idx], marker='o', color="green")
    plt.draw()


def plot_estimation(mean, std, title, value):
    x = np.linspace(mean - std * 4, mean + std * 4, 100)
    plt.title(title)
    plt.plot(x, mlab.normpdf(x, mean, std),
             label="Predicted probability:\n mu={}, sigma={}".format(int(mean), int(std)))
    plt.axvline(x=value, label="Actual value: {}".format(value), color='r')
    plt.ylim([0, 0.004])
    plt.legend()
