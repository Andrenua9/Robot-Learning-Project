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


def autolabel(rects,ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')



def main():
    #log_folder = "logs/walker2d_source_deep_udr30_seed42/"
    #plot_results(log_folder, title="Learning Curve Smoothed", algo="walker2d_source_deep_udr30_seed42")
    # Dati Baseline (Sostituisci con i tuoi valori medi)
    archs = ['Small', 'Medium', 'Deep']
    source_baseline = [2752.53 , 1056.84   , 1065.40] # Esempio: Performance ottime in training
    target_baseline = [ 1617.04 ,  511.01 , 626.23]    # Esempio: Crollo totale nel target
    source_std = [19.31 ,226.81, 156.90]         # Deviazioni standard per le barre di errore
    target_std = [405.24,  99.35,158.35]
    x = np.arange(len(archs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    rects1 = ax.bar(x - width/2, source_baseline, width, yerr=source_std, label='Source Env (Training)', color='#4e79a7', edgecolor='black')
    rects2 = ax.bar(x + width/2, target_baseline, width, yerr=target_std, label='Target Env (Transfer)', color='#e15759', edgecolor='black')

    ax.set_ylabel('Average Reward', fontweight='bold')
    ax.set_title('Reality Gap Analysis: UDR 30%', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    autolabel(rects1,ax)
    autolabel(rects2,ax)
    plt.tight_layout()
    plt.savefig(f"paper_figures/net_arch_ud0_baseline.pdf", format='pdf', dpi=300)

    plt.show()


if __name__ == "__main__":
    main()