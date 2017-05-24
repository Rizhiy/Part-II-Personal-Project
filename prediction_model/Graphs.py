import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, gamma


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


def plot_gamma(k, theta, title="Predicted probability density", value=None, legend=True):
    expected = k * theta
    if expected < 10:
        upper_bound = 20
    elif expected > 100:
        upper_bound = 800
    else:
        upper_bound = 200
    x = np.linspace(0, upper_bound, 100)
    plt.title(title)
    plt.plot(x, gamma.pdf(x, a=k, scale=theta),
             label=r"Predicted probability density:\\ k={}, $\theta$={}".format(int(k), int(theta)))
    if value is not None:
        plt.axvline(x=value, label="Actual value: {}".format(value), color='r')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    if legend:
        plt.legend()
