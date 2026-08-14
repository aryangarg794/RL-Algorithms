import matplotlib.pyplot as plt
import numpy as np


def plot_results(data, ax, timesteps, label, xlabel, ylabel, title, color='red', q=95, compare=None):
    data = np.asarray(data)
    num_seeds, num_datapoints = data.shape
    means = np.mean(data, axis=0)
    timesteps = np.linspace(0, timesteps, num_datapoints)

    lower_bound = np.percentile(data, q=100-q, axis=0)
    upper_bound = np.percentile(data, q=q, axis=0)
    ax.plot(timesteps, means, label=label, color=color, linewidth=2)
    ax.fill_between(
        timesteps, 
        lower_bound, 
        upper_bound, 
        color=color, 
        alpha=0.2, 
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    return ax