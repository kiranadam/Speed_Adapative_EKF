import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import resample as scipy_resample

SECONDS_2_TIMESTAMP = 1_000_000_000 # 1E9 nanoseconds == 1 second


def _interpolate_linear(x, y, x_new):
    """
    Copied from scipy
    Interpolate a function linearly given a new x.
    :param x: the original x of the function
    :param y: the original y of the function
    :param x_new: the new x to interpolate to
    :return: interpolated y-values on x_new
    """

    # 1. convert to numpy arrays if necessary.
    x = np.asarray(x)
    y = np.asarray(y)
    if len(y) == 0 or len(x) == 0:
        return y

    y_shape = y.shape
    y = y.reshape((y.shape[0], -1))

    # 2. Find where in the original data, the values to interpolate
    #    would be inserted.
    #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
    x_new_indices = np.searchsorted(x, x_new)

    # 3. Clip x_new_indices so that they are within the range of
    #    x indices and at least 1.  Removes mis-interpolation
    #    of x_new[n] = x[0]
    x_new_indices = x_new_indices.clip(1, len(x)-1).astype(int)

    # 4. Calculate the slope of regions that each x_new value falls in.
    lo = x_new_indices - 1
    hi = x_new_indices

    x_lo = x[lo]
    x_hi = x[hi]
    y_lo = y[lo]
    y_hi = y[hi]

    # Note that the following two expressions rely on the specifics of the
    # broadcasting semantics.
    slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

    # 5. Calculate the actual value for each entry in x_new.
    y_new = slope*(x_new - x_lo)[:, None] + y_lo

    return y_new.reshape(x_new.shape + y_shape[1:])

def interpolate(timestamps, data, fs):
    if len(timestamps) <= 1 or fs <= 0:
        return timestamps, data

    first_stamp = timestamps[0]

    srate = SECONDS_2_TIMESTAMP / fs
    xnew = np.arange(first_stamp, timestamps[-1], srate, dtype=np.int64)

    data = _interpolate_linear(timestamps, data, xnew)

    return xnew, data


def interpolate_stamps(timestamps, data, target_timestamps):
    """
    Interpolates given data and timestamps to new target timestamps

    Parameters
    ----------
    timestamps : np.array
        1-dimensional Numpy array with nanosecond timestamps
    data : np.array
        multi-dimensional Numpy array with the data to interpolate
    Returns
    -------
    np.array
        multi-dimensional Numpy array with the data values interpolated to the target timestamps 
    """
    if len(timestamps) <= 1:
        return data

    first_val = data[0]
    last_val = data[-1]

    interpolated_data = _interpolate_linear(timestamps, data, target_timestamps)

    interpolated_data[target_timestamps < timestamps[0]] = first_val
    interpolated_data[target_timestamps > timestamps[-1]] = last_val

    return interpolated_data



def high_pass(timestamps, data, fs, cutoff_fs):
    """
    High Pass filter. Removes frequencies <= cuttof frequency (i.e. passes frequencues higher than the cutoff)

    Parameters
    ----------
    timestamps : np.array
        1-dimensional Numpy array with nanosecond timestamps
    data : np.array
        multi-dimensional Numpy array with the data to filter
    fs : float
        sampling rate of the input data to filter
    cutoff_fs : 
        cutoff frequency to filter
    Returns
    -------
    np.array
        multi-dimensional Numpy array with the filtered data values
    """
    if len(timestamps) <= 1 or fs <= 0:
        return data

    wp = cutoff_fs / (fs / 2.0)
    b1, a1 = butter(2, wp, 'high')

    return filtfilt(b1, a1, data, padtype='constant', axis=0)


def low_pass(timestamps, data, fs, cutoff_fs):
    """
    Low Pass filter. Removes frequencies >= cuttof frequency (i.e. passes frequencues lower than the cutoff)

    Parameters
    ----------
    timestamps : np.array
        1-dimensional Numpy array with nanosecond timestamps
    data : np.array
        multi-dimensional Numpy array with the data to filter
    fs : float
        sampling rate of the input data to filter
    cutoff_fs : 
        cutoff frequency to filter
    Returns
    -------
    np.array
        multi-dimensional Numpy array with the filtered data values
    """

    if len(timestamps) <= 1 or fs <= 0:
        return data

    cutoff_fs = min(fs / 2.0, cutoff_fs)
    wp = cutoff_fs / (fs / 2.0)

    b1, a1 = butter(2, wp, 'low')

    return filtfilt(b1, a1, data, padtype='constant', axis=0)


def _nearest_pow2(x):
    """
    Finds nearest power of 2 number to a given number
    """
    if x == 0:
        return 0

    _ceil = math.ceil(math.log(x)/math.log(2))
    return int(math.pow(2, _ceil))


def _resample(timestamps, data, orig_fs, target_fs):
    if len(timestamps) <= 1 or orig_fs <= 0 or target_fs <= 0:
        return timestamps, data

    first_stamp = timestamps[0]

    ratio = float(target_fs) / orig_fs
    new_length = np.int64(np.ceil(len(timestamps) * ratio))
    pad_len = _nearest_pow2(len(timestamps)) - len(timestamps)
    interp_len = np.int64(np.ceil((len(timestamps) + pad_len) * ratio))

    padded = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')
    resampled = scipy_resample(padded, interp_len, axis=0)

    # remove extra padded values again
    resampled_data = resampled[:new_length]

    step_size = np.int64(SECONDS_2_TIMESTAMP / target_fs)
    timestamps = np.int64(first_stamp) + np.arange(start=0,
                                               stop=len(resampled_data) * step_size,
                                               step=step_size,
                                               dtype=np.int64)

    return timestamps, resampled_data


def _sort(timestamps, data):
    idx = np.argsort(timestamps)
    return timestamps[idx], data[idx]


def _deduplicate(timestamps, data):
    timestamps, idx = np.unique(timestamps, return_index=True)
    return timestamps, data[idx]


def resample(timestamps, data, target_fs):
    """
    resamples input data taking care of aliasing

    Parameters
    ----------
    timestamps : np.array
        1-dimensional Numpy array with nanosecond timestamps
    data : np.array
        multi-dimensional Numpy array with the data to resample
    target_fs : 
        target sampling rate
    Returns
    -------
    timestamps : np.array
        1-dimensional Numpy array with the timestamps corresponding to the resampled data
    data : np.array
        multi-dimensional Numpy array with the resampled data values
    """

    # Preprocess the data by sorting & removing duplicates
    timestamps, data = _sort(timestamps, data)
    timestamps, data = _deduplicate(timestamps, data)

    # Calculate the original sampling rate of the input data
    orig_fs = np.round(1.0 / (np.median(np.diff(timestamps)) / SECONDS_2_TIMESTAMP), 1)

    # Before we apply the Low Pass filter we will interpolate the data to a higher frequency.
    # By making sure the interpolation sampling rate is a power of 2 times the target sampling rate,
    # we make sure the inverse FFT in the resample method has a power of 2 input values as well.
    # FFTs are significantly faster with input size that equals a power of 2: O(n*log(n)) instead of O(n^2)
    interpolation_fs = max(2, _nearest_pow2(int(round(orig_fs / target_fs)))) * target_fs

    timestamps, data = interpolate(timestamps, data, interpolation_fs)
    data = low_pass(timestamps, data, interpolation_fs, target_fs / 2.0)
    timestamps, data = _resample(timestamps, data, interpolation_fs, target_fs)

    return timestamps, data


def sync_df(df, target_df):
    """
    Interpolates the data of the input dataframe to the timestamps of the target dataframe
    """
    timestamps = df.timestamp.values

    cols = df.columns.tolist()
    cols.remove('timestamp')
    data = df.loc[:, cols].values

    data = interpolate_stamps(timestamps, data, target_df.timestamp.values)
    values = np.hstack((timestamps.reshape((-1, 1)), data))

    return pd.DataFrame(values, columns=df.columns)


def resample_df(df, target_fs):
    """
    Resamples a dataframe to the target sampling rate
    """
    timestamps = df.timestamp.values

    cols = df.columns.tolist()
    cols.remove('timestamp')
    data = df.loc[:, cols].values

    timestamps, data = resample(timestamps, data, target_fs)
    values = np.hstack((timestamps.reshape((-1, 1)), data))

    return pd.DataFrame(values, columns=df.columns)


def high_pass_df(df, cutoff_fs):
    """
    Applies High Pass filter to a dataframe 
    """
    timestamps = df.timestamp.values

    cols = df.columns.tolist()
    cols.remove('timestamp')
    data = df.loc[:, cols].values

    orig_fs = np.round(1.0 / (np.diff(timestamps).mean() / SECONDS_2_TIMESTAMP), 1)

    data = high_pass(timestamps, data, orig_fs, cutoff_fs)

    values = np.hstack((timestamps.reshape((-1, 1)), data))

    return pd.DataFrame(values, columns=['timestamp'] + cols)

def low_pass_df(df, cutoff_fs):
    """
    Applies Low Pass filter to a dataframe 
    """
    timestamps = df.timestamp.values

    cols = df.columns.tolist()
    cols.remove('timestamp')
    data = df.loc[:, cols].values

    orig_fs = np.round(1.0 / (np.diff(timestamps).mean() / SECONDS_2_TIMESTAMP), 1)

    data = low_pass(timestamps, data, orig_fs, cutoff_fs)

    values = np.hstack((timestamps.reshape((-1, 1)), data))

    return pd.DataFrame(values, columns=['timestamp'] + cols)


def plot_sensor(df, name='', scale=1.0, ax=None, cols=None, base_ts=None):
    """
    Plots dataframe with sensor data 
    """
    if ax is None:
        f, ax = plt.subplots(1, figsize=(10, 5), sharex=True)

    if cols is None:
        cols = df.columns.tolist()
        cols.remove('timestamp')


    if base_ts is None:
        base_ts = df.timestamp.iat[0]

    ts = (df.timestamp - base_ts)/SECONDS_2_TIMESTAMP

    for col in cols:
        ax.plot(ts, (df.loc[:, col]) * scale, label=col)

    ax.legend()
    ax.set_ylabel(name)
    ax.set_xlabel('Time (s)')
