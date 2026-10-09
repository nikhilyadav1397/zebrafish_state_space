import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from external.CascadeTorch.cascade2p.utils import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.rcParams.update(
    {
        "font.size": 14,  # base font
        "axes.titlesize": 18,  # title
        "axes.labelsize": 16,  # x/y labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)


def plot_data_dist(raw_activity, plot_path="plots/raw_activity_distribution.png"):
    activity_values = raw_activity.ravel()
    activity_values = activity_values[np.isfinite(activity_values)]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        activity_values,
        bins=100,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.8,
    )

    ax.axvline(
        activity_values.mean(),
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {activity_values.mean():.3f}",
    )

    ax.axvline(
        np.median(activity_values),
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {np.median(activity_values):.3f}",
    )

    ax.set(
        xlabel="Activity value",
        ylabel="Number of observations",
        title="Distribution of raw activity values",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)


def plot_detrend(trial0_activity, trial0_activity_det, plot_path="plots"):
    act_mean = trial0_activity.mean(axis=0)
    act_std = trial0_activity.std(axis=0)
    detrended_mean = trial0_activity_det.mean(axis=0)
    detrended_std = trial0_activity_det.std(axis=0)

    time = np.arange(trial0_activity.shape[1])  # Assuming time is along axis 1

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(time, act_mean)
    axs[0].fill_between(time, act_mean - act_std, act_mean + act_std, alpha=0.3)
    axs[0].set_ylabel("Calcium activity")
    axs[0].set_title("Before detrending")

    axs[1].plot(time, detrended_mean)
    axs[1].fill_between(
        time, detrended_mean - detrended_std, detrended_mean + detrended_std, alpha=0.3
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Calcium activity")
    axs[1].set_title("After linear detrending")

    fig.suptitle("Trial 0: population activity")
    fig.tight_layout()
    fig.savefig(
        f"{plot_path}/trial0_detrending_comparison.png", dpi=300, bbox_inches="tight"
    )


def plot_neuron_activity_distribution(
    normalized_activity,
    mean_plot_path="plots/neuron_mean_activity_distribution.png",
    sd_plot_path="plots/neuron_activity_std_distribution.png",
):
    """
    Plot distributions of temporal mean and standard deviation
    across neurons.

    Parameters
    ----------
    normalized_activity : ndarray
        Array with shape (n_neurons, n_frames).
    plot_path : str or Path
        Directory in which figures are saved.

    Returns
    -------
    neuron_mean_activity : ndarray
        Temporal mean for each neuron.
    neuron_activity_std : ndarray
        Temporal standard deviation for each neuron.
    """
    normalized_activity = np.asarray(
        normalized_activity,
        dtype=float,
    )

    if normalized_activity.ndim != 2:
        raise ValueError(
            "normalized_activity must have shape " "(n_neurons, n_frames)."
        )

    neuron_mean_activity = np.nanmean(
        normalized_activity,
        axis=1,
    )
    neuron_activity_std = np.nanstd(
        normalized_activity,
        axis=1,
    )

    # ------------------------------------------------------------
    # Mean activity
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        neuron_mean_activity[np.isfinite(neuron_mean_activity)],
        bins=50,
    )

    ax.set_xlabel(r"Mean activity")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Distribution of mean activity across neurons")

    fig.tight_layout()
    fig.savefig(
        mean_plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Activity standard deviation
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        neuron_activity_std[np.isfinite(neuron_activity_std)],
        bins=50,
    )

    ax.set_xlabel(r"SD of activity")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Distribution of activity variability across neurons")

    fig.tight_layout()
    fig.savefig(
        sd_plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return neuron_mean_activity, neuron_activity_std


def plot_mean_std_relationship(
    normalized_activity,
    plot_path="plots/mean_std_distribution.png",
):
    """
    Plot the relationship between temporal mean and standard
    deviation across neurons.

    Parameters
    ----------
    normalized_activity : ndarray
        Array with shape (n_neurons, n_frames).
    plot_path : str or Path
        Directory in which the figure is saved.

    Returns
    -------
    neuron_stats : DataFrame
        Mean, variance and standard deviation for each neuron.
    """
    normalized_activity = np.asarray(
        normalized_activity,
        dtype=float,
    )

    if normalized_activity.ndim != 2:
        raise ValueError(
            "normalized_activity must have shape " "(n_neurons, n_frames)."
        )

    neuron_stats = pd.DataFrame(
        {
            "neuron_id": np.arange(normalized_activity.shape[0]),
            "mean": np.nanmean(normalized_activity, axis=1),
            "variance": np.nanvar(normalized_activity, axis=1),
            "std": np.nanstd(normalized_activity, axis=1),
        }
    )

    finite_mask = np.isfinite(neuron_stats["mean"]) & np.isfinite(neuron_stats["std"])

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        neuron_stats.loc[finite_mask, "mean"],
        neuron_stats.loc[finite_mask, "std"],
        alpha=0.3,
        s=10,
    )

    ax.set_xlabel(r"Mean activity ($\Delta F/F_0$)")
    ax.set_ylabel(r"SD of activity ($\Delta F/F_0$)")
    ax.set_title("Mean–SD relationship across neurons")

    fig.tight_layout()
    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return neuron_stats


def plot_random_traces(
    data,
    activity_cols,
    *,
    n_neurons=5,
    fs=3,
    seed=None,
    ylim=(-1.5, 4),
    plot_path="plots/random_activity_traces.png",
):
    """
    Plot random neurons in rows, with two different trials
    from the same neuron shown in the two columns.
    """

    rng = np.random.default_rng(seed)

    # Only neurons having at least two different trials are eligible
    trial_counts = data.groupby("neuron_id")["trial"].nunique()
    eligible_neurons = trial_counts[trial_counts >= 2].index.to_numpy()

    if n_neurons > len(eligible_neurons):
        raise ValueError(
            f"Requested {n_neurons} neurons, but only "
            f"{len(eligible_neurons)} have at least two trials."
        )

    selected_neurons = rng.choice(
        eligible_neurons,
        size=n_neurons,
        replace=False,
    )

    fig, axs = plt.subplots(
        n_neurons,
        2,
        figsize=(12, 2.5 * n_neurons),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    time = np.arange(len(activity_cols)) / fs

    for row_index, neuron_id in enumerate(selected_neurons):
        neuron_data = data.loc[data["neuron_id"] == neuron_id]

        available_trials = neuron_data["trial"].unique()
        selected_trials = rng.choice(
            available_trials,
            size=2,
            replace=False,
        )

        for column_index, trial in enumerate(selected_trials):
            ax = axs[row_index, column_index]

            selected_row = neuron_data.loc[neuron_data["trial"] == trial].iloc[0]

            trace = selected_row.loc[activity_cols].to_numpy(dtype=float)

            ax.plot(
                time,
                trace,
                color="steelblue",
                linewidth=1,
            )

            ax.set_title(f"Neuron {int(neuron_id)}, trial {int(trial)}")

            if column_index == 0:
                ax.set_ylabel(r"$\Delta F/F$")

    # Add x-axis labels to the bottom row
    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        "Two trials from randomly selected neurons",
        fontsize=14,
    )
    fig.tight_layout()

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )


def plot_random_traces_act(
    activity,
    *,
    n_neurons=5,
    fs=3,
    seed=None,
    plot_path="plots/random_activity_traces.png",
):
    """Plot continuous calcium traces from randomly selected neurons.

    Parameters
    ----------
    activity : array-like, shape (n_neurons, n_frames)
        Continuous calcium activity.
    n_neurons : int, default=5
        Number of neurons to plot.
    fs : float, default=3
        Sampling frequency in Hz.
    seed : int or None
        Random seed for reproducible neuron selection.
    plot_path : str or Path
        Location at which the figure is saved.

    Returns
    -------
    selected_neurons : ndarray
        Indices of the plotted neurons.
    """

    activity = np.asarray(activity, dtype=float)

    if activity.ndim != 2:
        raise ValueError("activity must have shape (n_neurons, n_total_frames).")

    if fs <= 0:
        raise ValueError("fs must be greater than zero.")

    n_total_neurons, n_total_frames = activity.shape

    if not 1 <= n_neurons <= n_total_neurons:
        raise ValueError(
            f"n_neurons must be between 1 and {n_total_neurons}; "
            f"received {n_neurons}."
        )

    rng = np.random.default_rng(seed)
    selected_neurons = rng.choice(
        n_total_neurons,
        size=n_neurons,
        replace=False,
    )

    time = np.arange(n_total_frames) / fs

    fig, axes = plt.subplots(
        nrows=n_neurons,
        ncols=1,
        figsize=(14, 2.1 * n_neurons),
        sharex=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes)

    for ax, neuron_id in zip(axes, selected_neurons):
        trace = activity[neuron_id]

        ax.plot(
            time,
            trace,
            color="#2878B5",
            linewidth=0.8,
            rasterized=True,
        )

        # Reference line showing the trace median
        median = np.nanmedian(trace)
        ax.axhline(
            median,
            color="black",
            linewidth=0.7,
            linestyle="--",
            alpha=0.35,
        )

        ax.set_ylabel(
            rf"Neuron {neuron_id}",
            fontsize=12,
        )

        ax.grid(
            axis="x",
            color="0.85",
            linewidth=0.6,
            alpha=0.7,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both",
            labelsize=8,
            direction="out",
        )

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    axes[-1].set_xlim(time[0], time[-1])

    fig.suptitle(
        "Continuous calcium activity",
        fontsize=16,
    )

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return selected_neurons


def plot_random_activity_with_spikes(
    normalized_activity,
    spike_prob,
    *,
    n_neurons=8,
    fs=3,
    seed=None,
    start_time=0,
    duration=300,
    plot_path="plots/random_activity_with_spikes.png",
):
    """
    Plot calcium activity and CASCADE inference in separate columns.

    Left column:
        Normalized calcium activity.

    Right column:
        CASCADE output for the same neuron.

    Both input arrays must have shape (n_neurons, n_frames).
    """
    normalized_activity = np.asarray(
        normalized_activity,
        dtype=float,
    )

    if hasattr(spike_prob, "detach"):
        spike_prob = spike_prob.detach().cpu().numpy()
    else:
        spike_prob = np.asarray(
            spike_prob,
            dtype=float,
        )

    if normalized_activity.ndim != 2:
        raise ValueError(
            "normalized_activity must have shape " "(n_neurons, n_frames)."
        )

    if spike_prob.shape != normalized_activity.shape:
        raise ValueError(
            "spike_prob and normalized_activity must have "
            f"the same shape, but received "
            f"{spike_prob.shape} and "
            f"{normalized_activity.shape}."
        )

    n_total_neurons, n_frames = normalized_activity.shape

    if n_neurons > n_total_neurons:
        raise ValueError("n_neurons cannot exceed the number of available neurons.")

    rng = np.random.default_rng(seed)

    selected_neurons = rng.choice(
        n_total_neurons,
        size=n_neurons,
        replace=False,
    )

    start_frame = int(round(start_time * fs))
    stop_frame = min(
        start_frame + int(round(duration * fs)),
        n_frames,
    )

    if start_frame < 0:
        raise ValueError("start_time must be non-negative.")

    if start_frame >= n_frames:
        raise ValueError(
            f"start_time={start_time} s lies outside the "
            f"{n_frames / fs:.1f} s recording."
        )

    if stop_frame <= start_frame:
        raise ValueError("duration must be greater than zero.")

    time = (
        np.arange(
            start_frame,
            stop_frame,
        )
        / fs
    )

    fig, axes = plt.subplots(
        n_neurons,
        2,
        figsize=(15, 2.5 * n_neurons),
        sharex=True,
        squeeze=False,
    )

    for row, neuron_id in enumerate(selected_neurons):
        calcium_ax = axes[row, 0]
        spike_ax = axes[row, 1]

        calcium_trace = normalized_activity[
            neuron_id,
            start_frame:stop_frame,
        ]

        inferred_spikes = spike_prob[
            neuron_id,
            start_frame:stop_frame,
        ]

        # --------------------------------------------------------
        # Left column: calcium trace
        # --------------------------------------------------------
        calcium_ax.plot(
            time,
            calcium_trace,
            color="steelblue",
            linewidth=1,
        )

        calcium_ax.set_ylabel(rf"Neuron {neuron_id}" "\n" r"$\Delta F/F_0$")

        # --------------------------------------------------------
        # Right column: CASCADE inference
        # --------------------------------------------------------
        finite_spikes = np.isfinite(inferred_spikes)

        spike_ax.plot(
            time,
            inferred_spikes,
            color="darkorange",
            linewidth=1,
        )

        spike_ax.fill_between(
            time,
            0,
            inferred_spikes,
            where=finite_spikes,
            color="darkorange",
            alpha=0.35,
        )

        spike_ax.axhline(
            0,
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )

        spike_ax.set_ylabel("Inferred spike rate")

    # Column titles
    axes[0, 0].set_title(
        "Normalized calcium activity",
        fontsize=13,
    )

    axes[0, 1].set_title(
        "CASCADE inference",
        fontsize=13,
    )

    # Only the bottom row needs x-axis labels
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle(
        f"Calcium traces and inferred spiking activity "
        f"({start_time:.1f}–{stop_frame / fs:.1f} s)",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98),
    )

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return selected_neurons


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


def plot_random_traces_subplots(
    activity,
    *,
    n_neurons=5,
    fs=3,
    seed=None,
    plot_path="plots/random_activity_traces.png",
):
    """
    Plot complete continuous traces from randomly selected neurons.

    Parameters
    ----------
    activity : ndarray
        Calcium activity with shape
        (n_neurons, n_total_frames).
    """
    activity = np.asarray(activity, dtype=float)

    if activity.ndim != 2:
        raise ValueError("activity must have shape " "(n_neurons, n_total_frames).")

    n_total_neurons, n_total_frames = activity.shape

    if n_neurons > n_total_neurons:
        raise ValueError(
            f"Requested {n_neurons} neurons, but activity contains "
            f"only {n_total_neurons} neurons."
        )

    rng = np.random.default_rng(seed)

    selected_neurons = rng.choice(
        n_total_neurons,
        size=n_neurons,
        replace=False,
    )

    ncols = 2
    nrows = int(np.ceil(n_neurons / ncols))

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 3 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axs = axs.ravel()
    time = np.arange(n_total_frames) / fs

    for ax, neuron_id in zip(axs, selected_neurons):
        trace = activity[neuron_id]

        ax.plot(
            time,
            trace,
            color="steelblue",
            linewidth=0.8,
        )

        ax.set_title(f"Neuron {neuron_id}")
        ax.set_xlabel("Time (s)")
        # ax.set_ylabel(r"F")

    # Hide an unused subplot when n_neurons is odd
    for ax in axs[n_neurons:]:
        ax.set_visible(False)

    fig.suptitle(
        "Continuous traces from randomly selected neurons",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
