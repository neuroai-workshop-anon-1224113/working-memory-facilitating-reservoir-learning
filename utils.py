# -*- coding: utf-8 -*-

"""
    File name: utils.py
    Description: a set of function not directly related to experiments
    Python version: 3.6
"""

import time
import numpy as np
import george
from george.kernels import ExpSquaredKernel
from warnings import warn
import h5py


class Timer:
    def __init__(self):
        self.elapsed_time = 0

    def start(self):
        self.elapsed_time = time.time()

    def stop(self):
        self.elapsed_time = time.time() - self.elapsed_time
        print("Done after %.0f seconds = %.1f minutes" % (self.elapsed_time, self.elapsed_time / 60))


def relu(x):
    return np.maximum(x, 0.0)


def get_short_target(t, frequency=1.0 / 1000):
    """Short signal used in referenced papers."""
    return 1.3 * (np.sin(2.0 * np.pi * frequency * t) / 1.5 + np.sin(4.0 * np.pi * frequency * t) / 3.0 +
                  np.sin(6.0 * np.pi * frequency * t) / 9.0 + np.sin(8.0 * np.pi * frequency * t) / 3.0)


def get_long_target(t, frequency=1.0 / 10000):
    """A longer version of the short signal with an additional slow component."""
    return 0.5 * np.sin(2.0 * np.pi * frequency * t) + 0.5 * get_short_target(t)


def sample_gp_function(t, variance=1000, seed=0):
    """
    Sample a function from GP with zeros on the boundary.
    :param t:           an array of time points; left and right values will be zero
    :param variance     variance of the kernel
    :param seed:        random seed of the sampler (=of numpy)
    :return: a sample from a GP at points t, the first and the last values are zero
    """
    np.random.seed(seed)

    period = len(t)
    kernel = ExpSquaredKernel(variance)
    gp = george.GP(kernel)
    gp.compute(np.array([0, period - 1]))

    target = np.zeros(len(t))
    target[1:-1] = gp.sample_conditional(np.array([0, 0]), t[1:-1])

    np.random.seed(None)
    return target


def correlate_multidim(x, y):
    """
    Correlation of two periodic signals of the same length, computed with the circular shift.
    x could be multi-dimensional, in which case correlation is performed along the last axis.
    """
    y = np.hstack((y, y))[:-1]
    old_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    new_x = np.zeros(x.shape)

    for i in range(new_x.shape[0]):
        new_x[i] = np.correlate(x[i], y, mode='valid')

    return new_x.reshape(old_shape)


def cross_correlate_periodic(x, y):
    """
    Normalised cross correlation of two periodic signals of the same length, computed with the circular shift.
    x could be multi-dimensional, in which case correlation is performed along the last axis.
    """
    return correlate_multidim(x - np.expand_dims(np.mean(x, axis=-1), axis=-1), y - np.mean(y)) / (
            np.expand_dims(np.std(x, axis=-1), axis=-1) * np.std(y) * len(y))


def autocorrelate(x):
    """Computes autocorrelation of a 1d signal."""
    return np.correlate(x - np.mean(x), x - np.mean(x), mode='full')[len(x):] / np.std(x) ** 2


def print_hdf_attributes(name, obj):
    """
    Prints directories and files of an .hdf file.
    Usage: file.visititems(print_hdf_attributes)
    """
    print(name)
    for key, val in obj.attrs.items():
        print("%s: %s" % (key, val))


def reshape_rows_to_blocks(x):
    """Reshapes a matrix such that each row becomes a square block. sqrt(x.shape) must be [int, int]."""
    block_size = int(np.sqrt(x.shape[1]))
    n_blocks = int(np.sqrt(x.shape[0]))
    x = x.reshape((x.shape[0], block_size, block_size))

    x = np.block(np.split(x, x.shape[0]))
    return np.reshape(np.split(x, n_blocks, axis=-1), (block_size * n_blocks, block_size * n_blocks))


def compute_shifted_mse(data, targets):
    """
    Compute mean squared error with a temporal shift.

    Shift the signal to achieve better mse (as it might be naturally shifted due to imperfectly learnt signal).
    Matches the MSE from
    Hoerzer, G. M.; Legenstein, R. & Maass, W.
    Emergence of Complex Computational Structures From Chaotic Neural Networks
    Through Reward-Modulated Hebbian Learning. Cereb Cortex, 2012
    :param data:        input signal of the shape [..., len(targets)]
    :param targets:     target signal
    :return: MSE of the shape data.shape[:-1]
    """
    mse = np.zeros_like(data).T
    targets_shape = np.ones_like(data.shape)
    targets_shape[-1] = len(targets)

    for i in range(data.shape[-1]):
        mse[i] = np.mean((data - np.roll(targets, i).reshape(targets_shape)) ** 2, axis=-1).T

    return np.min(mse.T, axis=-1)


def compute_shifted_cross_correlation(data, targets):
    """
    Compute cross-correlation with a temporal shift.

    Shift the signal to achieve better correlation (as it might be naturally shifted due to imperfectly learnt signal) as in
    Hoerzer, G. M.; Legenstein, R. & Maass, W.
    Emergence of Complex Computational Structures From Chaotic Neural Networks
    Through Reward-Modulated Hebbian Learning. Cereb Cortex, 2012
    :param data:        input signal of the shape [..., len(targets)]
    :param targets:     target signal
    :return: Minimum normalised cross-correlation of the shape data.shape[:-1]
    """
    cross_corr = cross_correlate_periodic(data, targets)

    return np.max(cross_corr, axis=-1)


def compute_learning_error(file, seed, metric='cross_corr'):
    """
    Computes mean error for the 1d signal learning task.
    :param file:    name of the .hdf file
    :param seed:    random seed for which to compute mean error
    :param metric:  cross_corr (normalised) or mse (with shift)
    :return: mean error for each entry in the results (the last two axis are reduced)
    """
    target_values = file['seed%d/target_values' % seed][:]
    signal = file['seed%d/test_results' % seed][0, :, :]
    signal = signal.reshape((signal.shape[0], 50, len(target_values)))

    if metric == 'cross_corr':
        return compute_shifted_cross_correlation(signal, target_values)
    elif metric == 'mse':
        return compute_shifted_mse(signal, target_values)
    else:
        warn('Expected metric to be either cross_corr or mse. Returning mse.')
        return compute_shifted_mse(signal, target_values)


def compute_and_save_errors(signal_length='short'):
    """
    Computes and saves MSE for a recorded signal to an .hdf file.
    :param signal_length: 'short' or 'long', specifies files to load
    :return: None
    """
    file_prefix = 'recordings/%s' % signal_length

    data_hebb = h5py.File('%s_learning_hebb.hdf' % file_prefix, 'r')
    time_stamp = data_hebb['seed0/time_stamp'][:]

    error_hebb = np.zeros((len(time_stamp), 50, 50))

    if signal_length != 'long_2k':
        data_attractor = h5py.File('%s_learning_hebb_attractor.hdf' % file_prefix, 'r')
        error_attractor = np.zeros((len(time_stamp), 50, 50))

    for seed in range(50):
        error_hebb[:, seed, :] = compute_learning_error(data_hebb, seed)

        if signal_length != 'long_2k':
            error_attractor[:, seed, :] = compute_learning_error(data_attractor, seed)

    data_hebb.close()
    if signal_length != 'long_2k':
        data_attractor.close()

    file = h5py.File('%s_errors.hdf' % file_prefix, 'w')
    file.create_dataset('hebb', data=error_hebb.reshape((len(time_stamp), 50 * 50)))

    if signal_length != 'long_2k':
        file.create_dataset('attractor', data=error_attractor.reshape((len(time_stamp), 50 * 50)))

        if signal_length == 'short' or signal_length == 'long':
            baseline = 0
            data = h5py.File('%s_learning_force.hdf' % file_prefix, 'r')

            for seed in range(5):
                baseline = max(baseline, np.max(compute_learning_error(data, seed)))
            data.close()

            file.create_dataset('baseline', data=baseline)
    file.create_dataset('time_stamp', data=time_stamp)
    file.close()
