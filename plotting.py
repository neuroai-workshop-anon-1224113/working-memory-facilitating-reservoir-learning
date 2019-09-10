# -*- coding: utf-8 -*-

"""
    File name: plotting.py
    Description: a set of visualisation functions for experiment analysis
    Python version: 3.6
"""

import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('Running w/o a display')
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import seaborn as sns
import h5py
from utils import autocorrelate, compute_and_save_errors
from utils import get_short_target, get_long_target, sample_gp_function
sns.set(style="darkgrid")
sns.set(font_scale=2)


def animate_firing_rates(square_rates):
    """
    Shows an animation of rates recording on a square.
    :param square_rates: numpy array of the shape [time, side_length, side_length]
    :return: None
    """
    max_rate = np.amax(square_rates)
    fig = plt.figure()
    frames = []

    for i in range(square_rates.shape[0]):
        frames.append([plt.imshow(square_rates[i, :, :], interpolation='none',
                                  origin='lower', vmax=max_rate, animated=True)])
    ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True, repeat_delay=1000)
    plt.title('bump attractor')
    plt.colorbar()
    plt.show()


def plot_bump(square_rates, filename=None):
    """
    Plots a square of neuron activity at a particular time.
    :param square_rates:    a square numpy array
    :param filename:        where to save the image. If not specified, just shows the image
    :return: None
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    max_rate = np.amax(square_rates)
    plt.imshow(square_rates, interpolation='none',
               origin='lower', vmax=max_rate, animated=True, cmap=plt.get_cmap('plasma'))
    plt.axis([0, int(square_rates.shape[0]), 0, int(square_rates.shape[0])])
    plt.xticks(np.linspace(0, int(square_rates.shape[0]), int(square_rates.shape[0]) // 10 + 1).astype(int))
    plt.yticks(np.linspace(10, int(square_rates.shape[0]), int(square_rates.shape[0]) // 10).astype(int))

    plt.axis([0, 50, 0, 50])
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([10, 20, 30, 40, 50])

    cax = fig.add_axes([0.83, 0.15, 0.04, 0.75])
    plt.colorbar(cax=cax)
    plt.tight_layout()

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_bump_tracking(bump_trajectory, side_length, color='r', filename=None):
    """
    Plots trajectory on a square.
    :param bump_trajectory: a numpy array of the shape [time, 2]
    :param side_length: side length of the square
    :param color: a matplotlib-compatible color
    :param filename: where to save the image. If not specified, just shows the image
    :return: None
    """
    plt.subplots(figsize=(5, 5))
    plt.plot(bump_trajectory[:, 0], bump_trajectory[:, 1],
             'o', markerfacecolor=color, markersize=1.0)
    plt.axis([0.0, side_length, 0, side_length])

    plt.axis([0, 50, 0, 50])
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([10, 20, 30, 40, 50])

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_learning_curves(errors, time_stamp, legends, force_baseline=None, filename=None, colors=None,
                         n_test_periods=50):
    """
    Plots learning curves with standard deviation for several signal along with a baseline value.
    :param errors:              a list of [n_points, n_trials] arrays
    :param time_stamp:          time points at which recordings where done
    :param legends:             labels of the curves
    :param force_baseline:      a baseline value (with a label FORCE baseline), None for no plotting
    :param filename:            where to save the image. If not specified, just shows the image
    :param colors:              a list of matplotlib colors for the curves. Default is ['g', 'r']
    :param n_test_periods:      how many test periods to take into account (up to 50)
    :return: None
    """
    plt.subplots(figsize=(18, 10))

    if colors is None:
        colors = ['g', 'r', 'b']

    for i in range(len(legends)):
        current_errors = errors[i].reshape((len(time_stamp), 50, 50))
        current_errors = current_errors[:, :, :n_test_periods].reshape((len(time_stamp), -1))

        mean = current_errors.mean(axis=1)
        plt.plot(time_stamp, mean, label=legends[i], color=colors[i % len(colors)])
        # std = errors[i].std(axis=1) / np.sqrt(errors[i].shape[1])

        percentiles = np.percentile(current_errors, (5, 95), axis=1)
        plt.fill_between(time_stamp, percentiles[0], percentiles[1], alpha=0.1, color=colors[i])

        percentiles = np.percentile(current_errors, (25, 75), axis=1)
        plt.fill_between(time_stamp, percentiles[0], percentiles[1], alpha=0.2, color=colors[i])
        # plt.fill_between(time_stamp, mean - std, mean + std, alpha=0.2, color=colors[i])

    plt.ylabel('cross-correlation')
    plt.xlabel('number of periods')

    if force_baseline is not None:
        plt.plot(time_stamp, force_baseline * np.ones_like(time_stamp), 'b--', label='FORCE baseline')
    plt.legend()
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_readout_vectors(vectors, baseline_vec, filename=None):
    """
    Plots relative distance from a baseline vector (on a polar plot).
    :param vectors:         a [n_vectors, dimension] numpy array
    :param baseline_vec:    a [dimension] numpy array
    :param filename:        where to save the image. If not specified, just shows the image
    :return: None
    """
    angles = np.arccos(vectors.dot(baseline_vec) / np.linalg.norm(baseline_vec) / np.linalg.norm(vectors, axis=-1))
    relative_length = np.linalg.norm(vectors, axis=-1) / np.linalg.norm(baseline_vec)

    plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    plt.plot(angles, relative_length, 'o')

    plt.text(np.pi * 1.03, np.max(relative_length) * 0.9, 'relative distance')
    plt.text(np.pi * 1.44, np.max(relative_length) * 0.9, 'angle')

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_example_outputs(zhat, target_function, filename=None):
    """
    Plots noiseless output in the beginning and end of learning.
    :param zhat:            a [n_time_points, time] array of noiseless output
    :param target_function: the target function (one period)
    :param filename:        where to save the image. If not specified, just shows the image
    :return: None
    """
    period = len(target_function)
    plt.subplots(figsize=(18, 10))

    plt.plot(np.hstack((target_function, target_function)), label='target function', color='g')
    plt.plot(np.hstack((zhat[0, :period], zhat[-1, :period])), label='noiseless output', color='r')

    plt.ylim((np.min(target_function) * 1.3, np.max(target_function) * 1.3))
    plt.text(150, np.min(target_function), 'beginning')
    plt.text(period + 150, np.min(target_function), 'end of learning')
    plt.axvline(period, linestyle='--', color='k')
    plt.legend()
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_neuron_autocorrelation(activity, target_values, filename=None):
    """
    Plots the neuron's autocorrelation in the beginning and end of learning and contrasts in to the target
    function's autocorrelation.
    :param activity:        neuron firing rates
    :param target_values:   targe function
    :param filename: where to save the image. If not specified, just shows the image
    :return: None
    """
    period = len(target_values)
    n_periods = 10
    prolonged_targets = np.repeat(target_values[:, None], n_periods, axis=1).reshape(n_periods * period, order='F')
    target_autocorr = autocorrelate(prolonged_targets)

    plt.subplots(figsize=(18, 10))

    plt.plot(np.hstack((autocorrelate(activity[0, :(n_periods * period)]),
                        autocorrelate(activity[-1, :(n_periods * period)]))), 'g', label='neuron')
    plt.plot(np.hstack((target_autocorr, target_autocorr)), 'b--', alpha=0.7, label='target function')

    plt.text(150, np.min(target_autocorr) * 1.2, 'beginning')
    plt.text(n_periods * period + 150, np.min(target_autocorr) * 1.2, 'end of learning')
    plt.axvline(n_periods * period, linestyle='--', color='k')
    plt.yticks([])
    plt.ylim((np.min(target_autocorr) * 1.4, np.max(target_autocorr) * 1.2))
    plt.legend()
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_targets_fourier(filename=None):
    """Plots Fourier components of target functions for 1s, 10s and different complexity."""
    n_seeds = 50
    colors = ['g', 'b']

    for length in [1000, 10000]:
        plt.subplots(figsize=(18, 10))

        time = np.linspace(0, length - 1, length)
        n_components = 2 * (length // 100)

        for i, variance in enumerate([10000, 2000]):
            gp_fft= np.zeros((n_seeds, n_components))
            for seed in range(n_seeds):
                gp_fft[seed] = np.abs(np.fft.rfft(sample_gp_function(time, variance, seed)))[:n_components]
            percentiles = np.percentile(gp_fft, (5, 95), axis=0)

            plt.plot(gp_fft.mean(0), '-o', color=colors[i])
            plt.fill_between(np.arange(0, n_components), percentiles[0], percentiles[1],
                             alpha=0.2, color=colors[i], label=r'GP, $\sigma^2$=%d' % variance)

        if length == 1000:
            values = get_short_target(time)
        else:
            values = get_long_target(time)

        plt.plot(np.abs(np.fft.rfft(values))[:n_components], '-o', color='r', label='hand-picked')

        plt.xticks(np.linspace(0, n_components - 1, 20))

        plt.xlabel('component')
        plt.ylabel('magnitude')
        plt.legend()

        plt.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename + '_%ds' % (length // 1000))


def plot_mnist_bumps(n_digits=10, input_file='recordings/mnist_attractors.hdf', phase='train', filename=None,
                     to_plot='mean'):
    """Plots bumps for different digits on the same plot."""
    direction_vector = np.exp(2.j * np.pi * np.linspace(0, 49, 50) / 50)

    def compute_position(points):
        """Computes centre of 1d positions, converting a line to a circle on a complex plane."""
        return (np.angle(np.sum(direction_vector[points.astype('int')], axis=0)) / np.pi * 50 / 2.0) % 50

    file = h5py.File(input_file, 'r')

    n_epochs = file['n_%s' % phase][()]

    colormap = plt.get_cmap('Spectral')
    fig, ax = plt.subplots(figsize=(10, 10))

    mean_trajectory = np.zeros((n_epochs, file['/0/%s/0trajectory' % phase][:].shape[0], 2))

    for digit in range(n_digits):
        mean_trajectory.fill(0.0)

        for epoch in range(n_epochs):
            mean_trajectory[epoch] = file['/%d/%s/%dtrajectory' % (digit, phase, epoch)][:]

            if to_plot == 'start':
                plt.plot(file['/%d/%s/%dtrajectory' % (digit, phase, epoch)][:100, 0],
                         file['/%d/%s/%dtrajectory' % (digit, phase, epoch)][:100, 1],
                         'o', markerfacecolor=colormap(digit / 9.0), markersize=3.0, alpha=0.5)
            if to_plot == 'mean':
                plt.plot(file['/%d/%s/%dtrajectory' % (digit, phase, epoch)][:, 0],
                         file['/%d/%s/%dtrajectory' % (digit, phase, epoch)][:, 1],
                         'o', markerfacecolor=colormap(digit / 9.0), markersize=1.5)

        # if to_plot == 'mean':
        #     plt.plot(compute_position(mean_trajectory[:, :, 0]), compute_position(mean_trajectory[:, :, 1]),
        #              'o', markerfacecolor=colormap(digit / 9.0), markersize=0.0, label='%d' % digit)
    legend = []
    labels = []

    for digit in range(n_digits):
        legend.append(Line2D([0], [0], color=colormap(digit / 9.0), lw=4))
        labels.append('%d' % digit)

    ax.legend(legend, labels, loc='upper right')

    plt.axis([0, 50, 0, 50])
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([10, 20, 30, 40, 50])

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def plot_mnist_results(n_digits=10, input_file='recordings/mnist_performance.hdf', filename=None):
    """Plots drawings of the learnt digits."""
    data = h5py.File(input_file, 'r')
    results = data['seed42/test_results'][:].reshape((2, 50, 10, 1000))
    targets = data['seed42/target_values'][:]

    for digit in range(n_digits):
        plt.subplots(figsize=(5, 5))

        plt.plot(targets[digit, :, 0], targets[digit, :, 1], 'b', linewidth=5, label='target')

        mean_results = np.zeros((2, 1000))
        for i in range(50):
            plt.plot(results[0, i, digit, :], results[1, i, digit, :], 'go', alpha=0.2)
            mean_results += results[:, i, digit, :]

        mean_results /= 50
        plt.plot(mean_results[0], mean_results[1], 'ro', markersize=5, label='mean output')
        plt.xticks([])
        plt.yticks([])
        # plt.legend()
        plt.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename + '%d' % digit)
        plt.close()


def plot_mnist_inputs(n_digits=10, input_file='recordings/mnist_inputs.hdf', filename=None):
    """Plots inputs that are fed to the attractor layer to start it."""
    file = h5py.File(input_file, 'r')

    for digit in range(n_digits):
        for phase in ['train', 'test']:
            data = file['%d/%s' % (digit, phase)][:]
            image_shape = (int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1])))

            fig, axis = plt.subplots(nrows=int(np.ceil(data.shape[0] / 10)), ncols=10,
                                     figsize=(10, np.ceil(data.shape[0] / 10)))

            for i in range(data.shape[0]):
                image = axis.flatten()[i].imshow(data[i].reshape(image_shape), cmap=plt.get_cmap('plasma'))

            for ax in axis.flatten():
                ax.axis('off')

            # plt.subplot_tool()  # online adjusting of subplots
            fig.subplots_adjust(top=0.95, bottom=0.05, left=0.0, right=0.9, hspace=0.0,
                                wspace=0.03)

            cax = fig.add_axes([0.91, 0.1, 0.04, 0.8])  # left, bottom, width, height
            fig.colorbar(image, cax=cax)

            if filename is None:
                plt.show()
            else:
                plt.savefig(filename + '%d_%s' % (digit, phase))
            plt.close()

    file.close()


def plot_stochastic_updates(filename='pics/stochastic_updates', input_file_prefix='recordings/stochastic_updates_hebb'):
    """Plots the mean time and number of updates to reach a particular threshold in the stochastic updates task."""
    hebb_results = h5py.File('%s.hdf' % input_file_prefix, 'r')
    attractor_results = h5py.File('%s_attractor.hdf' % input_file_prefix, 'r')

    update_probs = np.linspace(0.1, 1.0, 10)
    n_periods = hebb_results['1.0/seed0/max_periods'][()]

    updates = np.zeros((2, 10, 50, n_periods))
    corr = np.zeros((2, 10, 50, n_periods))

    for data_ind, data in enumerate([hebb_results, attractor_results]):
        for prob_ind, prob in enumerate(update_probs):
            for seed in range(50):
                updates[data_ind, prob_ind, seed] = data['%.1f/seed%d/updates' % (prob, seed)][:]
                corr[data_ind, prob_ind, seed] = data['%.1f/seed%d/correlation' % (prob, seed)][:]

    hebb_results.close()
    attractor_results.close()

    updates_ratio = updates / 1000

    file_type_suffix = ['_hebb', '_hebb_attractor']
    file_var_suffix = ['_updates', '_corr']
    y_labels = [r'mean fraction of updates $\pm$ SEM', r'mean cross-correlation $\pm$ SEM']
    colormap = plt.get_cmap('Spectral')

    for result_ind, result in enumerate([updates_ratio, corr]):
        mean = result.mean(axis=-2)  # average over seeds
        std = result.std(axis=-2) / np.sqrt(result.shape[-2])

        for type_ind in range(2):
            plt.subplots(figsize=(18, 10))
            for prob_ind, prob in enumerate(update_probs):
                plt.plot(np.arange(1, n_periods + 1), mean[type_ind, prob_ind], color=colormap(prob),
                         label='p=%.1f' % prob)
                plt.fill_between(np.arange(1, n_periods + 1), mean[type_ind, prob_ind] - std[type_ind, prob_ind],
                                 mean[type_ind, prob_ind] + std[type_ind, prob_ind], alpha=0.2, color=colormap(prob))

            plt.ylabel(y_labels[result_ind])
            plt.xlabel('period')
            plt.legend(loc='lower right')
            plt.tight_layout()

            if filename is None:
                plt.show()
            else:
                plt.savefig(filename + file_type_suffix[type_ind] + file_var_suffix[result_ind])


def plot_rate_tuning(filename='pics/stochastic_updates_tuning',
                     input_file_prefix='recordings/hebb_attractor_rate_tuning_gp'):
    """Plots the mean time and number of updates to reach a particular threshold in the rate tuning task."""
    results = h5py.File('%s.hdf' % input_file_prefix, 'r')

    rate_multiplier = np.linspace(0.1, 1.0, 10)
    n_periods = results['1.0/seed0/max_periods'][()]

    updates = np.zeros((10, 50, n_periods))
    corr = np.zeros((10, 50, n_periods))

    for ind, multiplier in enumerate(rate_multiplier):
        for seed in range(50):
            updates[ind, seed] = results['%.1f/seed%d/updates' % (multiplier, seed)][:]
            corr[ind, seed] = results['%.1f/seed%d/correlation' % (multiplier, seed)][:]

    results.close()

    updates_ratio = updates / 1000

    file_var_suffix = ['_updates', '_corr']
    y_labels = [r'mean fraction of updates $\pm$ SEM', r'mean cross-correlation $\pm$ SEM']
    colormap = plt.get_cmap('Spectral')

    for result_ind, result in enumerate([updates_ratio, corr]):
        mean = result.mean(axis=-2)  # average over seeds
        std = result.std(axis=-2) / np.sqrt(result.shape[-2])

        plt.subplots(figsize=(18, 10))
        for ind, multiplier in enumerate(rate_multiplier):
            plt.plot(np.arange(1, n_periods + 1), mean[ind], color=colormap(multiplier),
                     label='p=%.1f' % multiplier)
            plt.fill_between(np.arange(1, n_periods + 1), mean[ind] - std[ind],
                             mean[ind] + std[ind], alpha=0.2, color=colormap(multiplier))

        plt.ylabel(y_labels[result_ind])
        plt.xlabel('period')
        plt.legend(loc='lower right')
        plt.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename + file_var_suffix[result_ind])


def plot_stochastic_updates_curves_comparison(period_to_plot=20, filename='pics/stochastic_updates_comparison',
                                              rate_input='recordings/hebb_attractor_rate_tuning_gp',
                                              stochastic_input='recordings/stochastic_updates_hebb_attractor'):
    """Compares two parameter-dependent plots at a particular period."""
    rate_results = h5py.File('%s.hdf' % rate_input, 'r')
    stochastic_results = h5py.File('%s.hdf' % stochastic_input, 'r')

    parameters = np.linspace(0.1, 1.0, 10)
    n_periods = rate_results['1.0/seed0/max_periods'][()]
    corr = np.zeros((2, 10, 50, n_periods))

    for data_ind, data in enumerate([rate_results, stochastic_results]):
        for ind, param in enumerate(parameters):
            for seed in range(50):
                corr[data_ind, ind, seed] = data['%.1f/seed%d/correlation' % (param, seed)][:]

    rate_results.close()
    stochastic_results.close()

    labels = ['rate', 'probability']
    colors = ['r', 'g']

    mean = corr.mean(axis=-2)  # average over seeds
    std = corr.std(axis=-2) / np.sqrt(corr.shape[-2])

    plt.subplots(figsize=(18, 10))
    for data_ind in range(2):
        plt.plot(parameters, mean[data_ind, :, period_to_plot - 1],
                 label=labels[data_ind], color=colors[data_ind])
        plt.fill_between(parameters, mean[data_ind, :, period_to_plot - 1] - std[data_ind, :, period_to_plot - 1],
                         mean[data_ind, :, period_to_plot - 1] + std[data_ind, :, period_to_plot - 1],
                         alpha=0.2, color=colors[data_ind])

    plt.ylabel(r'mean cross-correlation at period=%d $\pm$ SEM' % period_to_plot)
    plt.xlabel('parameter')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_weak_input(filename='pics/weak_input', input_file_prefix='recordings/weak_input/weak_attractor_input'):
    """Plots the mean time and number of updates to reach a particular threshold in the weak input task."""
    results = h5py.File('%s_0.hdf' % input_file_prefix, 'r')
    n_periods = results['1.0/seed0/max_periods'][()]
    results.close()

    input_multiplier = np.linspace(0.0, 1.0, 11)

    corr = np.zeros((11, 50, n_periods))

    for ind in range(len(input_multiplier)):
        results = h5py.File('%s_%d.hdf' % (input_file_prefix, ind), 'r')

        for seed in range(50):
            corr[ind, seed] = results['1.0/seed%d/correlation' % seed][:]

        results.close()

    colormap = plt.get_cmap('Spectral')

    mean = corr.mean(axis=-2)  # average over periods
    std = corr.std(axis=-2) / np.sqrt(corr.shape[-2])

    plt.subplots(figsize=(18, 10))
    for ind, multiplier in enumerate(input_multiplier):
        plt.plot(np.arange(1, n_periods + 1), mean[ind], color=colormap(multiplier),
                 label='c=%.1f' % multiplier)
        plt.fill_between(np.arange(1, n_periods + 1), mean[ind] - std[ind],
                         mean[ind] + std[ind], alpha=0.2, color=colormap(multiplier))

    plt.ylabel(r'mean cross-correlation $\pm$ SEM')
    plt.xlabel('period')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_learning_comparison_experiment_curves(n_test_periods=50, file_suffix=''):
    for signal_length in ['short', 'long']:
        file = h5py.File('recordings/%s_errors.hdf' % signal_length, 'r')
        plot_learning_curves([file['hebb'][:], file['attractor'][:]], file['time_stamp'][:],
                             ['Hebb', 'Hebb + attractor'], file['baseline'][()],
                             'pics/%s_learning_curve%s' % (signal_length, file_suffix), n_test_periods=n_test_periods)
        file.close()

    for signal_length in ['short_gp', 'long_gp']:
        file = h5py.File('recordings/%s_errors.hdf' % signal_length, 'r')
        plot_learning_curves([file['hebb'][:], file['attractor'][:]], file['time_stamp'][:],
                             ['Hebb', 'Hebb + attractor'], None,
                             'pics/%s_learning_curve%s' % (signal_length, file_suffix), n_test_periods=n_test_periods)
        file.close()


def plot_learning_comparison_experiment():
    # bump plots
    bump_trajectory = np.load('recordings/first_task_bump_trajectory.npy')
    plot_bump_tracking(bump_trajectory, side_length=50, filename='pics/first_task_bump_trajectory.png')

    bump_rates = np.load('recordings/first_task_bump_rates.npy')
    plot_bump(bump_rates[500].reshape((50, 50)), filename='pics/first_task_bump.png')

    # functions' Fourier transforms
    plot_targets_fourier(filename='pics/target_fourier')

    # learning curves
    for signal_length in ['short', 'long', 'short_gp', 'long_gp']:
        compute_and_save_errors(signal_length)

    plot_learning_comparison_experiment_curves(50, file_suffix='')
    plot_learning_comparison_experiment_curves(1, file_suffix='_one_test')

    # example performance
    for input_suffix in ['', '_attractor']:
        for signal_length in ['short', 'long', 'short_gp', 'long_gp']:
            file = h5py.File('recordings/%s_learning_hebb%s.hdf' % (signal_length, input_suffix), 'r')
            plot_example_outputs(file['seed42/test_results'][0], file['seed42/target_values'][:],
                                 filename='pics/output_%s_learning_hebb%s' % (signal_length, input_suffix))
            if signal_length == 'short' or signal_length == 'long':
                plot_neuron_autocorrelation(file['seed42/test_results'][1], file['seed42/target_values'][:],
                                            filename='pics/autocorr_%s_learning_hebb%s' % (signal_length, input_suffix))
            file.close()

    # # weights convergence
    # for input_suffix in ['', '_attractor']:
    #     for signal_length in ['short', 'long']:
    #         file = h5py.File('recordings/%s_learning_hebb%s.hdf' % (signal_length, input_suffix), 'r')
    #         file_force = h5py.File('recordings/%s_learning_force%s.hdf' % (signal_length, input_suffix), 'r')
    #
    #         final_vectors = np.zeros((50, 1000))
    #
    #         for seed in range(50):
    #             final_vectors[seed, :] = file['seed%d/output_weights' % seed][-1, 0, :]
    #
    #         plot_readout_vectors(final_vectors, file_force['seed42/output_weights'][-1, 0, :],
    #                              filename='pics/weights_%s_learning_hebb%s' % (signal_length, input_suffix))
    #
    #         file.close()
    #         file_force.close()


def plot_stochastic_updates_experiment():
    plot_stochastic_updates(filename='pics/stochastic_updates',
                            input_file_prefix='recordings/stochastic_updates_hebb')
    plot_stochastic_updates(filename='pics/stochastic_updates_hard',
                            input_file_prefix='recordings/stochastic_updates_hard_hebb')
    plot_rate_tuning(filename='pics/stochastic_updates_tuning',
                     input_file_prefix='recordings/hebb_attractor_rate_tuning_gp')
    plot_stochastic_updates_curves_comparison(period_to_plot=20, filename='pics/stochastic_updates_comparison',
                                              rate_input='recordings/hebb_attractor_rate_tuning_gp',
                                              stochastic_input='recordings/stochastic_updates_hebb_attractor')
    plot_stochastic_updates_curves_comparison(period_to_plot=100, filename='pics/stochastic_updates_comparison_end',
                                              rate_input='recordings/hebb_attractor_rate_tuning_gp',
                                              stochastic_input='recordings/stochastic_updates_hebb_attractor')


def plot_mnist_experiment():
    # MNIST bumps
    # inputs
    plot_mnist_inputs(input_file='recordings/mnist_inputs.hdf', filename='pics/mnist_inputs')
    # mean trajectory
    plot_mnist_bumps(phase='train', filename='pics/mnist_train_bumps', input_file='recordings/mnist_attractors.hdf',
                     to_plot='mean')
    plot_mnist_bumps(phase='test', filename='pics/mnist_test_bumps', input_file='recordings/mnist_attractors.hdf',
                     to_plot='mean')
    # start
    plot_mnist_bumps(phase='train', filename='pics/mnist_train_bumps_start',
                     input_file='recordings/mnist_attractors.hdf',
                     to_plot='start')
    plot_mnist_bumps(phase='test', filename='pics/mnist_test_bumps_start', input_file='recordings/mnist_attractors.hdf',
                     to_plot='start')
    # results
    plot_mnist_results(filename='pics/mnist_result', input_file='recordings/mnist_performance.hdf')
    plot_mnist_results(filename='pics/mnist_force_result', input_file='recordings/mnist_force_performance.hdf')

    # Ideal bumps
    # inputs
    plot_mnist_inputs(input_file='recordings/ideal_digit_inputs.hdf', filename='pics/ideal_digit_inputs')
    # mean trajectory
    plot_mnist_bumps(phase='train', filename='pics/ideal_digit_train_bumps',
                     input_file='recordings/ideal_digit_attractors.hdf', to_plot='mean')
    plot_mnist_bumps(phase='test', filename='pics/ideal_digit_test_bumps',
                     input_file='recordings/ideal_digit_attractors.hdf', to_plot='mean')
    # start
    plot_mnist_bumps(phase='train', filename='pics/ideal_digit_train_bumps_start',
                     input_file='recordings/ideal_digit_attractors.hdf', to_plot='start')
    plot_mnist_bumps(phase='test', filename='pics/ideal_digit_test_bumps_start',
                     input_file='recordings/ideal_digit_attractors.hdf', to_plot='start')
    # results
    plot_mnist_results(filename='pics/ideal_digit_result', input_file='recordings/ideal_digit_performance.hdf')


def plot_weak_input_experiments():
    plot_weak_input(filename='pics/weak_input',
                    input_file_prefix='recordings/weak_input/weak_attractor_input')
    plot_weak_input(filename='pics/weak_input_hard',
                    input_file_prefix='recordings/weak_input/weak_attractor_input_hard')


def plot_delayed_updates_experiment():
    plot_weak_input(filename='pics/delayed_updates',
                    input_file_prefix='recordings/delayed_updates/delayed_updates')
    plot_weak_input(filename='pics/delayed_updates_hard',
                    input_file_prefix='recordings/delayed_updates/delayed_updates_hard')


def main():
    plot_learning_comparison_experiment()
    plot_stochastic_updates_experiment()
    plot_mnist_experiment()
    plot_weak_input_experiments()
    plot_delayed_updates_experiment()


if __name__ == '__main__':
    main()
