import numpy as np
from scipy.signal import detrend
from plotting import *
import h5py
from scipy.ndimage import gaussian_filter1d


def percentile_norm(
    raw_activity,
    neuron_ids,
    percentile=20,
):
    raw_activity = np.asarray(raw_activity, dtype=float)
    neuron_ids = np.asarray(neuron_ids)

    F0 = np.empty((raw_activity.shape[0], 1), dtype=float)
    for neuron_id in np.unique(neuron_ids):
        neuron_mask = neuron_ids == neuron_id

        # One scalar from all trials and time points of this neuron
        neuron_F0 = np.percentile(
            raw_activity[neuron_mask].ravel(),
            percentile,
        )
        if np.isclose(neuron_F0, 0):
            raise ValueError(f"F0 is approximately zero for neuron {neuron_id}.")
        F0[neuron_mask] = neuron_F0
    normalized_activity = (raw_activity - F0) / F0
    return F0, normalized_activity


def load_mat_file(file_path):
    """
    Load a .mat file and return the data and info as numpy arrays.
    """
    mat = h5py.File(file_path, "r")
    data = mat["DATA"]["grp"]["input"]
    info = mat["DATA"]["grp"]["info"]
    data = np.array(data)
    n_chunks = data.shape[1] // 300  # 45
    n_neurons = data.shape[0]
    data = pd.DataFrame(data.reshape(-1, 300))
    data.insert(0, "neuron_id", np.repeat(np.arange(n_neurons), n_chunks))
    data.insert(1, "trial", np.tile(np.arange(n_chunks), n_neurons))
    data = data.astype(np.float64)
    return data, np.array(info)


def detrend_activity(
    data,
    activity_cols,
    axis=1,
    detrend_type="linear",
):
    activity = data.loc[:, activity_cols].to_numpy(dtype=float)

    if not np.isfinite(activity).all():
        raise ValueError("Activity contains NaN or infinite values.")

    detrended_activity = detrend(
        activity,
        axis=axis,
        type=detrend_type,
    )

    # Compare original and detrended activity for trial 0
    trial0_mask = data["trial"].eq(0).to_numpy()

    trial0_activity = activity[trial0_mask]
    trial0_activity_detrended = detrended_activity[trial0_mask]

    plot_detrend(
        trial0_activity,
        trial0_activity_detrended,
    )

    # Preserve neuron and trial identifiers
    detrended_data = data.copy()
    detrended_data.loc[:, activity_cols] = detrended_activity

    return detrended_data


def smooth_cascade_output(
    spike_prob,
    *,
    fs=3,
    sigma_seconds=1 / 3,
):
    spike_prob = np.asarray(
        spike_prob,
        dtype=float,
    )

    smoothed = np.full_like(
        spike_prob,
        np.nan,
    )

    sigma_frames = sigma_seconds * fs

    for neuron_index, trace in enumerate(spike_prob):
        finite = np.isfinite(trace)

        if finite.any():
            smoothed[neuron_index, finite] = gaussian_filter1d(
                trace[finite],
                sigma=sigma_frames,
                mode="nearest",
            )

    return smoothed
