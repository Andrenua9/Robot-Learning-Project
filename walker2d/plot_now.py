import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve", algo=""):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    :param algo: (str) the algorithm name for filename
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    
    # Save plot to plots directory
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/learning_curve_{algo}.png")
    plt.show()
    plt.close(fig)

def main():
    log_folder = "logs/walker2d_source_medium_udr30_seed42/"
    plot_results(log_folder, title="Learning Curve Smoothed", algo="learning_curve_walker2d_source_medium_udr30_seed42")

if __name__ == "__main__":
    main()