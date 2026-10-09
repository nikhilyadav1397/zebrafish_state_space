import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from scipy.signal import detrend
from pathlib import Path
from external.CascadeTorch.cascade2p import cascade
from external.CascadeTorch.cascade2p.utils import *
from src.plotting import *
from src.functions import *
import torch


def plot_noise_level_distribution(traces, frame_rate):
    """
    Plots a histogram of the noise levels across all neurons in the dataset

    """
    try:
        import seaborn as sns

        sns.set()
        plt.style.use("seaborn-darkgrid")
    except:
        pass

    noise_levels = calculate_noise_levels(traces, frame_rate)

    percent999 = np.nanpercentile(noise_levels, 99.9)

    plt.figure(1121)
    plt.hist(noise_levels, density=True, bins=100)
    # plt.xlim([0, percent999])
    plt.xlabel("Noise level (% s^(1/2))")
    plt.title("Histogram of noise levels across neurons")

    return noise_levels


if __name__ == "__main__":
    Path("plots").mkdir(exist_ok=True)
    fs = 3  # Frequency of the data in Hz
    time = np.arange(300) / fs  # time for one trial in seconds

    # Load the .mat file
    with h5py.File("./data/fbos7_OB_testDATA.mat", "r") as mat:
        raw_activity = np.asarray(
            mat["DATA"]["grp"]["input"],
            dtype=np.float64,
        )
        info = np.asarray(mat["DATA"]["grp"]["info"])

    n_neurons, n_frames = raw_activity.shape

    plot_random_traces_subplots(
        raw_activity,
        n_neurons=8,
        fs=fs,
        seed=42,
        plot_path="plots/random_raw_activity_traces.png",
    )

    noise_levels = plot_noise_level_distribution(
        raw_activity,
        fs,
    )

    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(
        "plots/cascade_noise_level_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Inspect raw and normalized continuous activity
    # ------------------------------------------------------------
    plot_data_dist(raw_activity, plot_path="plots/raw_activity_distribution.png")

    # ------------------------------------------------------------
    # Distribution and mean–SD diagnostics
    # ------------------------------------------------------------
    plot_neuron_activity_distribution(
        raw_activity,
        mean_plot_path="plots/neuron_mean_activity_distribution.png",
        sd_plot_path="plots/neuron_activity_std_distribution.png",
    )

    plot_mean_std_relationship(
        raw_activity, plot_path="plots/mean_std_distribution.png"
    )
