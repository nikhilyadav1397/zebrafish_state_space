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

    # ------------------------------------------------------------
    # Normalize each neuron using baseline mean
    # ------------------------------------------------------------

    F0 = np.nanmean(
        raw_activity[:, 80:150],
        axis=1,
        keepdims=True,
    )

    normalized_activity = (raw_activity - F0) / (F0)

    plot_random_traces_subplots(
        normalized_activity,
        n_neurons=8,
        fs=fs,
        seed=42,
        plot_path="plots/random_normalized_activity_traces.png",
    )

    noise_levels = plot_noise_level_distribution(
        normalized_activity,
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
    # Distribution and mean–SD diagnostics
    # ------------------------------------------------------------
    plot_neuron_activity_distribution(
        normalized_activity,
        mean_plot_path="plots/neuron_mean_activity_distribution.png",
        sd_plot_path="plots/neuron_activity_std_distribution.png",
    )

    plot_mean_std_relationship(
        normalized_activity, plot_path="plots/mean_std_distribution.png"
    )

    # ------------------------------------------------------------
    # CASCADE spike inference
    # ------------------------------------------------------------

    n_cascade_neurons = min(1000, normalized_activity.shape[0])

    cascade_activity = normalized_activity[:n_cascade_neurons]

    model_name = "Global_EXC_3Hz_smoothing400ms_high_noise"
    total_array_size = cascade_activity.itemsize * cascade_activity.size * 64 / 1e9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    spike_prob = cascade.predict(
        model_name,
        cascade_activity,
        model_folder="external/CascadeTorch/Pretrained_models",
        device=device,
        verbosity=1,
    )

    firing_rates = smooth_cascade_output(
        spike_prob,
        fs=fs,
        sigma_seconds=1 / 3,
    )

    plot_data_dist(firing_rates, plot_path="plots/firing_rates_dist.png")

    # firing_rates = np.sqrt(firing_rates)

    # plot_data_dist(firing_rates, plot_path="plots/firing_rates_dist_sqrt.png")

    neuron_std = np.nanstd(
        firing_rates,
        axis=1,
        keepdims=True,
    )
    neuron_mean = np.nanmean(
        firing_rates,
        axis=1,
        keepdims=True,
    )
    safe_std = np.where(
        neuron_std > 1e-8,
        neuron_std,
        1.0,
    )

    zscored_activity = firing_rates - neuron_mean

    selected_neurons = plot_random_activity_with_spikes(
        cascade_activity,
        zscored_activity,
        n_neurons=6,
        fs=fs,
        seed=42,
        start_time=0,
        duration=300,
        plot_path="plots/random_activity_with_spikes_mean.png",
    )
