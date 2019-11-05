
Learn more or give us feedback
# -*- coding: utf-8 -*-

"""
    File name: experiments.py
    Description: a set of functions for recording experiments
    Author: Roman Pogodin
    Python version: 3.6
"""

import simulation_interface as sim
import numpy as np
import xml.etree.ElementTree as xml_parser
import re
import matplotlib.pyplot as plt
from utils import get_long_target, get_short_target, sample_gp_function
import pandas as pd
from utils import compute_learning_error
import h5py


def record_attractor():
    """Attractor recording for the first set of experiments; 10s; default parameters."""
    recorder = sim.AttractorRecorder(randseed=42)
    recorder.record_output(10000, 'recordings/first_task_bump')


def record_one_signal_experiments(length='short', n_seeds=50):
    """
    Records all experiments for a short/long 1d signal signal.

    This function runs several trials of the Hebb/Hebb with attractor input and a single trial
    of the FORCE learning, recording each of them to a corresponding .hdf file.
    The variable learning_pattern has a format [[n_1, m_1], ...] where n_1 is the number of train periods between
    each 50 periods test and m_1 is the maximum number of train periods for which this pattern is done.
    Consequently, the very last element m_k sets the overall number of train periods.

    The network's state is memorised after each train period and recovered after testing.

    Note that there is no test for an untrained network as the weighs are initialised to 0 and produce the same error
    every time
    """
    if length == 'short':
        period = 1000
        target_func = get_short_target
    elif length == 'long':
        period = 10000
        target_func = get_long_target

    file_prefix = 'recordings/%s_learning' % length

    recorded_variables = ['zhat', 'x[0]', 'x[10]', 'x[25]']
    learning_pattern = np.array([[10, 50], [25, 200], [100, 400]])
    attractor_input = np.load('recordings/first_task_bump_output.npy')

    for randseed in range(n_seeds):
        print("Recording for Hebb learning + attractor input, seed=%d" % randseed)
        sim.record_learning_dynamics(period, 50, target_func, 1000, attractor_input, 'hebb', recorded_variables,
                                     learning_pattern, '%s_hebb_attractor.hdf' % file_prefix, randseed)

        print("Recording for Hebb learning, seed=%d" % randseed)
        sim.record_learning_dynamics(period, 50, target_func, 1000, None, 'hebb', recorded_variables,
                                     learning_pattern, '%s_hebb.hdf' % file_prefix, randseed)

        # if length == 'long':
        #     print("Recording for Hebb learning 2k, seed=%d" % randseed)
        #     sim.record_learning_dynamics(period, 50, target_func, 2000, None, 'hebb', recorded_variables,
        #                                  learning_pattern, '%s_hebb2k.hdf' % file_prefix, randseed)

    for randseed in range(5):
        print("Recording for FORCE")
        sim.record_learning_dynamics(period, 50, target_func, 1000, None, 'force', recorded_variables,
                                     learning_pattern, '%s_force.hdf' % file_prefix, randseed)


def tune_hebb_parameters(length='long', n_seeds=10, gp=False):
    """Finds an optimal set of parameters for the Hebbian rule"""
    if length == 'short':
        period = 1000
        target_func = get_short_target
    elif length == 'long':
        period = 10000
        target_func = get_long_target

    if gp:
        target_func = None
    target_values = None

    # the defaults are 5 * 1e-4 and 20 * 1e3
    learning_rates = 5.0 * np.logspace(-5, -3, 3)
    learning_decays = 20.0 * np.logspace(2, 4, 3)
    learning_pattern = np.array([[400, 400]])
    recorded_variables = ['zhat']

    results = pd.DataFrame(np.zeros((len(learning_decays) * len(learning_rates), 4)),
                           columns=['rate', 'decay', 'mean', 'std'])

    if gp:
        file_prefix = 'recordings/tuning/%s_gp_tuning_hebb' % length
    else:
        file_prefix = 'recordings/tuning/%s_tuning_hebb' % length

    num_trial = 0  # parameters are saved in the hdf file
    for rate in learning_rates:
        for decay in learning_decays:
            print("Recording trial number %d" % num_trial)

            correlations = np.zeros((n_seeds, 50))
            for randseed in range(n_seeds):
                if gp:
                    target_values = sample_gp_function(np.arange(period), variance=10000, seed=randseed)

                sim.record_learning_dynamics(period, 50, target_func, 1000, None, 'hebb', recorded_variables,
                                             learning_pattern, '%s_%d.hdf' % (file_prefix, num_trial), randseed,
                                             rate, decay, target_values=target_values)

                file = h5py.File('%s_%d.hdf' % (file_prefix, num_trial), 'r')
                correlations[randseed] = compute_learning_error(file, randseed)
                file.close()

            results.iloc[num_trial, :] = (rate, decay, correlations.flatten().mean(), correlations.flatten().std())
            num_trial += 1

    results.to_csv(file_prefix + '.csv')


def convert_svg2array(input_file, output_file=None, show_result=False):
    """Reads an SVG and converts its trajectory to a 2d array."""
    tree = xml_parser.parse(input_file)
    root = tree.getroot()

    data = np.array(root[-1][0].attrib['d'].split(' '))
    reg_exp = re.compile('.*,.*')
    index = np.array([bool(reg_exp.match(data[i])) for i in range(len(data))])
    data = data[index]
    print('Length of the signal from %s is %d' % (input_file, len(data)))

    trajectory = np.zeros((len(data), 2))
    for i in range(len(data)):
        trajectory[i] = data[i].split(',')

    # get rid of relative coordinates
    for i in range(1, len(trajectory)):
        trajectory[i][0] += trajectory[i - 1][0]
        trajectory[i][1] += trajectory[i - 1][1]

    trajectory[:, 0] = (trajectory[:, 0] - trajectory[0, 0])
    trajectory[:, 1] = -(trajectory[:, 1] - trajectory[0, 1])
    trajectory /= np.amax(np.abs(trajectory))

    if output_file is not None:
        np.save(output_file, trajectory)
    
    if show_result:
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-')
        plt.xlim((np.amin(trajectory) * 1.1, np.amax(trajectory) * 1.1))
        plt.ylim((np.amin(trajectory) * 1.1, np.amax(trajectory) * 1.1))
        plt.show()

    return trajectory


def get_digit_drawings(period=1000):
    """Loads svg drawings of digits and converts them to 2d trajectories of the particular period."""
    data = np.zeros((10, period, 2))

    for i in range(10):
        trajectory = convert_svg2array('data_to_learn/%d.svg' % i)

        if trajectory.shape[0] > period:
            print("Trajectory length=%d is greater than the period=%d, truncating" % (trajectory.shape[0], period))

        data[i, :, 0] = np.interp(np.arange(period), np.linspace(0, period - 1, trajectory.shape[0]), trajectory[:, 0])
        data[i, :, 1] = np.interp(np.arange(period), np.linspace(0, period - 1, trajectory.shape[0]), trajectory[:, 1])

    return data


def record_mnist_experiment():
    """Records the experiments with leaning digit drawings with MNIST-induced bumps."""
    sim.record_mnist_inputs()
    sim.record_mnist_attractors()

    sim.record_ideal_digit_inputs()
    sim.record_mnist_attractors(filename='recordings/ideal_digit_attractors.hdf',
                                input_file='recordings/ideal_digit_inputs.hdf')

    targets = get_digit_drawings()

    sim.record_mnist_learning_dynamics(targets, learning_rule='hebb', external_input_type='mnist',
                                       filename='recordings/mnist_performance.hdf', n_epochs=4)
    sim.record_mnist_learning_dynamics(targets, learning_rule='hebb', external_input_type='ideal',
                                       filename='recordings/ideal_digit_performance.hdf', n_epochs=4)
    sim.record_mnist_learning_dynamics(targets, learning_rule='force', external_input_type='mnist',
                                       filename='recordings/mnist_force_performance.hdf', n_epochs=4)


def record_stochastic_updates_experiment(file_prefix='recordings/stochastic_updates_hebb', variance=10000):
    """Records experiments with probabilistic updates and corresponding stability of Hebbian learning."""
    attractor_input = np.load('recordings/first_task_bump_output.npy')

    max_periods = 100
    hebb_rate = 0.0005

    for randseed in range(50):
        target_values = sample_gp_function(np.arange(1000), variance=variance, seed=randseed)

        for update_prob in np.linspace(0.1, 1.0, 10):
            sim.record_constant_rate_updates(target_values=target_values, update_prob=update_prob,
                                             attractor_input=None, learning_rule='hebb',
                                             max_periods=max_periods,
                                             filename='%s.hdf' % file_prefix,
                                             randseed=randseed, hebb_rate=hebb_rate)

            sim.record_constant_rate_updates(target_values=target_values, update_prob=update_prob,
                                             attractor_input=attractor_input, learning_rule='hebb',
                                             max_periods=max_periods,
                                             filename='%s_attractor.hdf' % file_prefix,
                                             randseed=randseed, hebb_rate=hebb_rate)


def record_constant_rate_tuning_experiment(file_prefix='recordings/hebb_attractor_rate_tuning_gp.hdf', variance=10000):
    """Records experiments with probabilistic updates and corresponding stability of Hebbian learning."""
    attractor_input = np.load('recordings/first_task_bump_output.npy')

    max_periods = 100
    hebb_rate = 0.0005

    for randseed in range(50):
        target_values = sample_gp_function(np.arange(1000), variance=variance, seed=randseed)

        for rate_multiplier in np.linspace(0.1, 1.0, 10):
            sim.record_constant_rate_updates(target_values=target_values, update_prob=1.0,
                                             attractor_input=attractor_input, learning_rule='hebb',
                                             max_periods=max_periods,
                                             filename='%s' % file_prefix,
                                             randseed=randseed, hebb_rate=hebb_rate * rate_multiplier,
                                             hdf_folder=rate_multiplier)


def record_gp_signal_experiments(length='short', n_seeds=50):
    """
    Records all experiments for a short/long 1d signal signal drawn from a GP.

    This function runs several trials of the Hebb/Hebb with attractor input and a single trial
    of the FORCE learning, recording each of them to a corresponding .hdf file.
    The variable learning_pattern has a format [[n_1, m_1], ...] where n_1 is the number of train periods between
    each 50 periods test and m_1 is the maximum number of train periods for which this pattern is done.
    Consequently, the very last element m_k sets the overall number of train periods.

    The network's state is memorised after each train period and recovered after testing.

    Note that there is no test for an untrained network as the weighs are initialised to 0 and produce the same error
    every time.
    """
    if length == 'short':
        period = 1000
    elif length == 'long':
        period = 10000

    file_prefix = 'recordings/%s_gp_learning' % length

    recorded_variables = ['zhat']
    learning_pattern = np.array([[10, 50], [25, 200], [100, 400]])
    attractor_input = np.load('recordings/first_task_bump_output.npy')

    for randseed in range(n_seeds):
        target_values = sample_gp_function(np.arange(period), variance=10000, seed=randseed)

        print("Recording for Hebb learning + attractor input, seed=%d" % randseed)
        sim.record_learning_dynamics(period, 50, None, 1000, attractor_input, 'hebb', recorded_variables,
                                     learning_pattern, '%s_hebb_attractor.hdf' % file_prefix, randseed,
                                     target_values=target_values)

        print("Recording for Hebb learning, seed=%d" % randseed)
        sim.record_learning_dynamics(period, 50, None, 1000, None, 'hebb', recorded_variables,
                                     learning_pattern, '%s_hebb.hdf' % file_prefix, randseed,
                                     target_values=target_values)


def record_weak_attractor_input_experiment(file_prefix='recordings/weak_attractor_input', variance=10000,
                                           delayed_updates=False):
    """Records experiments with weakened attractor input and corresponding stability of Hebbian learning."""
    attractor_input = np.load('recordings/first_task_bump_output.npy')

    max_periods = 100
    hebb_rate = 0.0005

    for randseed in range(50):
        target_values = sample_gp_function(np.arange(1000), variance=variance, seed=randseed)

        for attractor_multiplier in np.linspace(0.0, 1.0, 11):
            sim.record_constant_rate_updates(target_values=target_values, update_prob=1.0,
                                             attractor_input=attractor_multiplier * attractor_input,
                                             learning_rule='hebb',
                                             max_periods=max_periods,
                                             filename='%s_%d.hdf' % (file_prefix, int(attractor_multiplier * 10)),
                                             randseed=randseed, hebb_rate=hebb_rate, delayed_updates=delayed_updates)


def main():
    # Experiments for the first part (1-dimensional signals).
    record_attractor()

    tune_hebb_parameters('long', n_seeds=10, gp=False)
    tune_hebb_parameters('long', n_seeds=10, gp=True)

    record_one_signal_experiments('short')
    record_one_signal_experiments('long')  # this will take some time

    # GP functions
    record_gp_signal_experiments(length='short')
    record_gp_signal_experiments(length='long')

    # MNIST experiments
    record_mnist_experiment()

    # Stochastic updates
    record_stochastic_updates_experiment('recordings/stochastic_updates_hebb', variance=10000)
    record_constant_rate_tuning_experiment(file_prefix='recordings/hebb_attractor_rate_tuning_gp.hdf', variance=10000)
    record_stochastic_updates_experiment('recordings/stochastic_updates_hard_hebb', variance=2000)

    # Weak attractor input
    record_weak_attractor_input_experiment(file_prefix='recordings/weak_input/weak_attractor_input', variance=10000)
    record_weak_attractor_input_experiment(file_prefix='recordings/weak_input/weak_attractor_input_hard', variance=2000)

    # Delayed updates
    record_weak_attractor_input_experiment(file_prefix='recordings/delayed_updates/delayed_updates', variance=10000,
                                           delayed_updates=True)
    record_weak_attractor_input_experiment(file_prefix='recordings/delayed_updates/delayed_updates_hard', variance=2000,
                                           delayed_updates=True)


if __name__ == '__main__':
    main()
