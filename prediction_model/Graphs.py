import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


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
