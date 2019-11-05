# -*- coding: utf-8 -*-

"""
    File name: simulation_interface.py
    Description: a set of functions for recording and training/testing neural networks
    Author: Roman Pogodin
    Python version: 3.6
"""

IS_REPRODUCIBLE = True

from warnings import warn
import numpy as np

# For reproducibility in Keras
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
if IS_REPRODUCIBLE:
    warn('Tensorflow will be running on a single CPU to get a reproducible result. '
         'Set IS_REPRODUCIBLE to False to use more cores or GPUs')
    import tensorflow as tf
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                  device_count={'CPU': 1, 'GPU': 0})
    from keras import backend as K
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

import reservoir_net
import attractor_net
import h5py
import keras
import keras.datasets.mnist as mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, Conv1D, MaxPooling2D, Activation, Reshape
from keras.optimizers import RMSprop
from keras.models import Model
from keras.utils import plot_model
from utils import compute_shifted_cross_correlation
from utils import reshape_rows_to_blocks


class AttractorRecorder:
    """
    Records and saves a simulation of the attractor network
    """
    def __init__(self, n_attractor_neurons=2500, n_reservoir_neurons=1000, randseed=42, noisy_weights_std=2.0):
        self._n_attractor_neurons = n_attractor_neurons
        self._n_reservoir_neurons = n_reservoir_neurons
        self._attractor = attractor_net.AttractorNetwork(noisy_weights_std=noisy_weights_std, randseed=randseed,
                                                         num_rec=self._n_attractor_neurons)
        self._warm_up_time = 100

        weights_shape = (self._n_reservoir_neurons, self._n_attractor_neurons)
        connect_prob = 0.1

        rand_generator = np.random.RandomState(1)
        weights_indicator = np.array(rand_generator.uniform(size=weights_shape) <= connect_prob, dtype=int)
        std = np.sqrt(1.0 / (connect_prob * self._n_reservoir_neurons))
        self._output_weights = rand_generator.normal(loc=0.0, scale=std, size=weights_shape) * weights_indicator

    def record_output(self, max_time, filename=None, warm_up_input=None, hdf_file=None, hdf_prefix=''):
        """
        Records a bump and saves its trajectory, activity and converted activity (that is, multiplied
        by a random Gaussian weight matrix of the shape (_n_reservoir_neurons, _n_attractor_neurons)
        """
        bump_trajectory = np.zeros((max_time, 2))
        converted_output = np.zeros((max_time, self._n_reservoir_neurons))
        network_rates = np.zeros((max_time, self._n_attractor_neurons))
        self._attractor.reset_network()

        for net_time in range(self._warm_up_time):
            self._attractor.update(warm_up_input)

        for net_time in range(max_time):
            self._attractor.update()
            bump_trajectory[net_time] = self._attractor.position
            converted_output[net_time] = self._output_weights.dot(self._attractor.firing_rates)
            network_rates[net_time] = self._attractor.firing_rates

        if filename is not None:
            np.save(filename + '_rates', network_rates)  # requires a lot of space
            np.save(filename + '_output', converted_output)
            np.save(filename + '_trajectory', bump_trajectory)

        if hdf_file is not None:
            # hdf_file.create_dataset('%srates' % hdf_prefix, data=network_rates,
            #                         compression='lzf', shuffle=True)   # requires a lot of space
            hdf_file.create_dataset('%soutput' % hdf_prefix, data=converted_output,
                                    compression='lzf', shuffle=True)
            hdf_file.create_dataset('%strajectory' % hdf_prefix, data=bump_trajectory,
                                    compression='lzf', shuffle=True)


class ReservoirInterface:
    """
    An interface for training and testing of a reservoir network.
    """
    def __init__(self, reservoir, period):
        self.reservoir = reservoir
        self._period = int(period)
        self._recorded_variables = []
        self._n_target_functions = 1
        self._warm_up_time = 1000  # for reset_network

    def reset_network(self):
        """Resets network and makes it run with no input for self._warm_up_time"""
        self.reservoir.reset_network_activity()

        for net_time in range(self._warm_up_time):
            self.reservoir.update()

    def _simulate_period(self, attractor_input=None, target_function=None, count_updates=False):
        """
        Runs the network for one period.
        :param attractor_input:     an already converted (of shape (n_rec,) attractor input or None
        :param target_function:     a target function for train or None for test
        :param count_updates:       whether to count the number of updates
        :return: values of each variable from self._recorded_variables of the shape [n_vars, time]
        """
        result = np.zeros((len(self._recorded_variables), self._period))
        n_updates = 0

        current_attractor_input = None
        current_target_function = None

        for net_time in range(self._period):
            if attractor_input is not None:
                current_attractor_input = attractor_input[net_time]
            if target_function is not None:
                current_target_function = target_function[net_time]

            self.reservoir.update(external_input=current_attractor_input, target_function=current_target_function)

            for i in range(len(self._recorded_variables)):
                result[i, net_time] = eval('self.reservoir.' + self._recorded_variables[i])

            if count_updates and self.reservoir.are_weights_updated():
                n_updates += 1

        return result, n_updates

    def _simulate(self, n_epochs, recorded_variables=None, period_reset=False,
                  attractor_input=None, target_functions=None, count_updates=False):
        """
        Simulates a network for several epochs.
        :param n_epochs:            number of periods for each target function
        :param recorded_variables:  a list of 1d variables to record
        :param period_reset:        whether to reset the net after each period
        :param attractor_input:     an already converted (of shape (period, n_rec) attractor input or None
        :param target_functions:    a list of target functions for train or None for test
        :param count_updates:       whether to count the number of updates
        :return: values of each variable from recorded_variables of the shape [n_vars, time]
        """
        if attractor_input is not None and len(attractor_input.shape) == 2:  # for one task
            attractor_input = attractor_input[None, :, :]

        if recorded_variables is None:
            self._recorded_variables = []
        else:
            self._recorded_variables = recorded_variables.copy()

        n_updates = 0

        result = np.zeros((len(self._recorded_variables), self._period * n_epochs * self._n_target_functions))

        current_attractor_input = None
        current_target_function = None

        for i in range(n_epochs * self._n_target_functions):
            if period_reset:
                self.reset_network()

            if attractor_input is not None:
                current_attractor_input = attractor_input[i % self._n_target_functions]
            if target_functions is not None:
                current_target_function = target_functions[i % self._n_target_functions]

            result[:, i * self._period:(i + 1) * self._period], curr_n_updates = \
                self._simulate_period(attractor_input=current_attractor_input,
                                      target_function=current_target_function,
                                      count_updates=count_updates)

            if target_functions is not None and self.reservoir.are_updates_delayed():
                self.reservoir.apply_weight_updates()
            n_updates += curr_n_updates

        return result, n_updates

    def train(self, n_epochs, target_functions, recorded_variables=None,
              period_reset=False, attractor_input=None, count_updates=False):
        """
        Trains the network for several epochs.
        :param n_epochs:            number of periods for each target function
        :param target_functions:    a list of target functions for train or None for test
        :param recorded_variables:  a list of 1d variables to record
        :param period_reset:        whether to reset the net after each period
        :param attractor_input:     an already converted (of shape (period, n_rec) attractor input or None
        :param count_updates:       whether to count the number of updates
        :return: values of each variable from recorded_variables of the shape [n_vars, time]
        """
        if len(target_functions.shape) == 1:  # for one task
            target_functions = target_functions[None, :]
        self._n_target_functions = target_functions.shape[0]

        return self._simulate(n_epochs, recorded_variables, period_reset, attractor_input,
                              target_functions, count_updates)

    def test(self, n_epochs, recorded_variables=None, period_reset=False, attractor_input=None):
        """
        Test the network for several epochs (no plasticity).
        :param n_epochs:                number of periods for each target function
        :param recorded_variables:      a list of 1d variables to record
        :param period_reset:            whether to reset the net after each period
        :param attractor_input:         an already converted (of shape (n_rec,) attractor input or None
        :return: values of each variable from recorded_variables of the shape [n_vars, time]
        """
        return self._simulate(n_epochs, recorded_variables, period_reset, attractor_input)[0]


def setup_reservoir(n_reservoir_neurons=1000, learning_rule='hebb', randseed=42, hebb_rate=0.0005,
                    hebb_decay=20.0 * 1e3, num_out=1, hebb_const=False, update_prob=1.0, delayed_updates=False):
    """Setups a reservoir network instance."""
    reservoir = reservoir_net.ReservoirNet(num_rec=n_reservoir_neurons, randseed=randseed, learning_rule=learning_rule,
                                           hebb_rate=hebb_rate, hebb_decay=hebb_decay, num_out=num_out,
                                           hebb_const=hebb_const, update_prob=update_prob,
                                           delayed_updates=delayed_updates)
    return reservoir


def run_reservoir(period=1000, num_train_epochs=10, num_test_epochs=50, target_values=None, n_reservoir_neurons=1000,
                  attractor_input=None, learning_rule='hebb', train_recordings=None, test_recordings=None,
                  simulator=None, count_updates=False, freeze_train_activity=False):
    """
    Trains and then tests a reservoir network.
    :param period:                  length of one period
    :param num_train_epochs:        number of train periods for each target function
    :param num_test_epochs:         number of test periods for each target function
    :param target_values:           array of target functions, shape=(n_function, period) (or (period,) for one target)
    :param n_reservoir_neurons:     number of reservoir neurons
    :param attractor_input:         an already converted (of shape (period, n_rec) attractor input or None
    :param learning_rule:           'hebb' for reward-modulated Hebbian rule or 'force' for FORCE
    :param train_recordings:        a list of 1d variables to record during training
    :param test_recordings:         a list of 1d variables to record during testing
    :param simulator:               an instance of ReservoirInterface
    :param count_updates:           whether to count the number of updates
    :param freeze_train_activity:   whether to memorise train activity before test and recover it after
    :return: values of each variable from recorded_variables of the shape [n_vars, time] for train and test
    """
    if simulator is None:
        reservoir = setup_reservoir(n_reservoir_neurons=n_reservoir_neurons, learning_rule=learning_rule)
        simulator = ReservoirInterface(reservoir, period)

    train_results, n_updates = simulator.train(num_train_epochs, target_values, train_recordings,
                                               attractor_input=attractor_input, count_updates=count_updates)

    if freeze_train_activity:
        simulator.reservoir.memorise_state()

    test_results = simulator.test(num_test_epochs, test_recordings,
                                  attractor_input=attractor_input)

    if freeze_train_activity:
        simulator.reservoir.recover_memorised_state()

    if count_updates:
        return train_results, test_results, n_updates
    return train_results, test_results


def compute_num_recordings(recording_pattern):
    """Calculates the total number of recordings for a recording pattern."""
    total_num_recordings = recording_pattern[0, 1] // recording_pattern[0, 0]  # +1 for init recording
    for i in range(1, recording_pattern.shape[0]):
        total_num_recordings += (recording_pattern[i, 1] - recording_pattern[i - 1, 1]) // \
                                recording_pattern[i, 0]
    return total_num_recordings


def record_learning_dynamics(period=1000, num_test_epochs=50, target_function=None, n_reservoir_neurons=1000,
                             attractor_input=None, learning_rule='hebb', test_recordings=None,
                             test_recording_pattern=np.array([[10, 50], [25, 200], [100, 400]]),  # train_periods, until
                             filename='test_performance.hdf', randseed=42, hebb_rate=0.0005, hebb_decay=20 * 1e3,
                             target_values=None):
    """
    Records dynamics of a reservoir network to an .hdf file.

    The variable test_recording_pattern has a format [[n_1, m_1], ...] where n_1 is the number of train periods between
    each 50 periods test and m_1 is the maximum number of train periods for which this pattern is done.
    Consequently, the very last element m_k sets the overall number of train periods.

    The network's state is memorised after each train period and recovered after testing.

    Note that there is no test for an untrained network as the weighs are initialised to 0 and produce the same error
    every time

    :param period:                  length of one period
    :param num_test_epochs:         number of test periods for each target function
    :param target_function:         a function object to evaluate a 1d function
    :param n_reservoir_neurons:     number of reservoir neurons
    :param attractor_input:         an already converted (of shape (period, n_rec) attractor input or None
    :param learning_rule:           'hebb' for reward-modulated Hebbian rule or 'force' for FORCE
    :param test_recordings:         a list of 1d variables to record during testing
    :param test_recording_pattern:  look above
    :param filename:                name of the file with recordings
    :param randseed:                random seed for the reservoir
    :param hebb_rate:               initial learning rate for Hebb learning
    :param hebb_decay:              hyperbolic decay for Hebb learning
    :param target_values:           if target function is not specified, these values are used
    :return: None
    """
    reservoir = setup_reservoir(n_reservoir_neurons=n_reservoir_neurons, learning_rule=learning_rule, randseed=randseed,
                                hebb_rate=hebb_rate, hebb_decay=hebb_decay)
    simulator = ReservoirInterface(reservoir, period)

    if target_function is not None:
        target_values = target_function(np.linspace(0, period - 1, period), 1.0 / period)
    elif target_values is None:
        warn('Target function is not specified, test only')

    n_finished_periods = 0
    n_pattern = 0

    total_num_recordings = compute_num_recordings(test_recording_pattern)

    results_shape = (len(test_recordings), total_num_recordings, num_test_epochs * period)
    weights_shape = (total_num_recordings, 1, n_reservoir_neurons)

    output_file = h5py.File(filename, 'a')
    output_file.create_dataset('/seed%d/recorded_variables' % randseed, data=np.array(test_recordings, dtype='S'))
    output_file.create_dataset('/seed%d/target_values' % randseed, data=target_values)
    output_file.create_dataset('/seed%d/hebb_rate' % randseed, data=hebb_rate)
    output_file.create_dataset('/seed%d/hebb_decay' % randseed, data=hebb_decay)
    test_results = output_file.create_dataset('/seed%d/test_results' % randseed, results_shape, 'float')
    output_weights = output_file.create_dataset('/seed%d/output_weights' % randseed, weights_shape, 'float')
    time_stamp = output_file.create_dataset('/seed%d/time_stamp' % randseed, (total_num_recordings,), 'int')
    output_file.flush()

    # # not needed as w(0)=0, so the error is always the same in the beginning
    # record_simulation_step(time_stamp, test_results, output_weights, 0, n_finished_periods, simulator,
    #                        period, 0, num_test_epochs, target_values, attractor_input, test_recordings)
    current_time = 0

    while n_finished_periods < test_recording_pattern[-1, -1]:
        if n_finished_periods >= test_recording_pattern[n_pattern, -1]:
            n_pattern += 1

        n_finished_periods += test_recording_pattern[n_pattern, 0]
        time_stamp[current_time] = n_finished_periods

        test_results[:, current_time, :] = run_reservoir(period, test_recording_pattern[n_pattern, 0],
                                                         num_test_epochs, target_values,
                                                         attractor_input=attractor_input,
                                                         test_recordings=test_recordings,
                                                         simulator=simulator, freeze_train_activity=True)[1]
        output_weights[current_time, :, :] = simulator.reservoir.weights_out.copy()

        current_time += 1
        output_file.flush()

    output_file.close()


def record_mnist_inputs(n_digits=10, n_train_epochs=50, n_test_epochs=50, filename='recordings/mnist_inputs.hdf'):
    """Takes first examples from the MNIST dataset and converts them to bump's inputs."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1) / 255.0
    y_train_categorical = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)

    file = h5py.File(filename, 'w')
    file.create_dataset('n_train', data=n_train_epochs)
    file.create_dataset('n_test', data=n_test_epochs)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), use_bias=False, input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    model.add(Dense(25))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.add(Reshape((25, 1)))
    model.add(Conv1D(25, 2, use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Activation("relu", name='attractor_input'))
    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    plot_model(model, to_file='mnist_model.pdf', show_shapes=True, rankdir='LR')

    history = model.fit(x_train, y_train_categorical,
                        batch_size=256,
                        epochs=20,
                        verbose=1,
                        validation_data=(x_test, y_test_categorical),
                        shuffle=False)

    score = model.evaluate(x_test, y_test_categorical, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    attractor_input = Model(inputs=model.input, outputs=model.get_layer('attractor_input').output)

    def scale_input(x, scaling=2):
        return np.kron(x, np.ones((scaling, scaling)))

    for digit in range(n_digits):
        train = attractor_input.predict(x_train[y_train == digit][:n_train_epochs])
        test = attractor_input.predict(x_test[y_test == digit][:n_test_epochs])

        train = train.reshape(n_train_epochs, int(np.sqrt(train.shape[1])), int(np.sqrt(train.shape[1])))
        test = test.reshape(n_test_epochs, int(np.sqrt(test.shape[1])), int(np.sqrt(test.shape[1])))

        for i in range(n_train_epochs):
            train[i] = reshape_rows_to_blocks(train[i])
        for i in range(n_test_epochs):
            test[i] = reshape_rows_to_blocks(test[i])

        train = scale_input(train, scaling=2).reshape((n_train_epochs, -1))
        test = scale_input(test, scaling=2).reshape((n_test_epochs, -1))

        train = 1 * (train / train.max(axis=1)[:, None])
        test = 1 * (test / test.max(axis=1)[:, None])

        file.create_dataset('/%d/train' % digit, data=train)
        file.create_dataset('/%d/test' % digit, data=test)
        file.flush()

    file.close()


def record_ideal_digit_inputs(n_digits=10, n_train_epochs=50, n_test_epochs=50,
                              filename='recordings/ideal_digit_inputs.hdf'):
    """Records inputs to the attractor layer that are noiseless and distinct for different digits."""
    file = h5py.File(filename, 'w')
    file.create_dataset('n_train', data=n_train_epochs)
    file.create_dataset('n_test', data=n_test_epochs)

    train = np.zeros((n_train_epochs, 50, 50))
    test = np.zeros((n_test_epochs, 50, 50))

    for digit in range(n_digits):
        train.fill(0.0)
        test.fill(0.0)

        train[:, int(5 * digit):int(5 * (digit + 1)), int(5 * digit):int(5 * (digit + 1))] = 1
        test[:, int(5 * digit):int(5 * (digit + 1)), int(5 * digit):int(5 * (digit + 1))] = 1

        file.create_dataset('/%d/train' % digit, data=train.reshape((n_train_epochs, 2500)))
        file.create_dataset('/%d/test' % digit, data=test.reshape((n_test_epochs, 2500)))
        file.flush()

    file.close()


def record_mnist_attractors(n_digits=10, signal_length=1000, filename='recordings/mnist_attractors.hdf',
                            input_file='recordings/mnist_inputs.hdf', randseed=0):
    """Records attractor dynamics for converted inputs from MNIST images."""
    input_file = h5py.File(input_file, 'r')
    output_file = h5py.File(filename, 'w')
    recorder = AttractorRecorder(n_attractor_neurons=2500, randseed=randseed, noisy_weights_std=6.0)

    n_train = input_file['n_train'][()]
    n_test = input_file['n_test'][()]

    output_file.create_dataset('n_train', data=n_train)
    output_file.create_dataset('n_test', data=n_test)

    for digit in range(n_digits):
        train_input = input_file['/%d/train' % digit]
        for epoch in range(n_train):
            recorder.record_output(signal_length, warm_up_input=train_input[epoch], hdf_file=output_file,
                                   hdf_prefix='/%d/train/%d' % (digit, epoch))
            output_file.flush()

        test_input = input_file['/%d/test' % digit]
        for epoch in range(n_test):
            recorder.record_output(signal_length, warm_up_input=test_input[epoch], hdf_file=output_file,
                                   hdf_prefix='/%d/test/%d' % (digit, epoch))
            output_file.flush()

    input_file.close()
    output_file.close()


def record_mnist_learning_dynamics(target_values, n_reservoir_neurons=1000, learning_rule='hebb',
                                   external_input_type='mnist', filename='./recordings/mnist_performance.hdf',
                                   randseed=42, hebb_rate=0.0005, hebb_decay=20 * 1e3, n_epochs=1):
    """
    Records test results for digits' drawings learning induced by pre-processed MNIST inputs/ideal noiseless inputs.
    :param target_values:           targets of the shape [n_targets, n_dimensions, period]
    :param n_reservoir_neurons:     number of reservoir neurons
    :param learning_rule:           hebb or force
    :param external_input_type:     mnist or ideal
    :param filename:                where to save the result
    :param randseed:                random seed for the reservoir
    :param hebb_rate:               learning rate
    :param hebb_decay:              time decay constant (rate decays as 1 / (1 + t/T))
    :param n_epochs:                number of repeats for the training input
    :return: None
    """
    reservoir = setup_reservoir(n_reservoir_neurons=n_reservoir_neurons, learning_rule=learning_rule, randseed=randseed,
                                hebb_rate=hebb_rate, hebb_decay=hebb_decay, num_out=2)
    test_recordings = ['zhat[0]', 'zhat[1]']

    input_file = h5py.File('./recordings/mnist_attractors.hdf', 'r')

    n_train = input_file['n_train'][()]
    n_test = input_file['n_test'][()]
    period = input_file['/0/train/0output'][:].shape[0]

    if external_input_type == 'ideal':
        input_file.close()
        input_file = h5py.File('./recordings/ideal_digit_attractors.hdf', 'r')

    if external_input_type == 'mnist' or external_input_type == 'ideal':
        train_input = np.zeros((10 * n_train, period, 1000))
        test_input = np.zeros((10 * n_test, period, 1000))

        for digit in range(10):
            for i in range(n_train):
                train_input[digit + i * 10] = input_file['/%d/train/%doutput' % (digit, i)][:]
            for i in range(n_test):
                test_input[digit + i * 10] = input_file['/%d/test/%doutput' % (digit, i)][:]
    else:
        train_input = None
        test_input = None

    simulator = ReservoirInterface(reservoir, period)

    output_file = h5py.File(filename, 'a')
    output_file.create_dataset('/seed%d/recorded_variables' % randseed, data=np.array(test_recordings, dtype='S'))
    output_file.create_dataset('/seed%d/target_values' % randseed, data=target_values)
    output_file.create_dataset('/seed%d/hebb_rate' % randseed, data=hebb_rate)
    output_file.create_dataset('/seed%d/hebb_decay' % randseed, data=hebb_decay)

    # just training
    for i in range(n_epochs):
        run_reservoir(period, n_train, 0, target_values, attractor_input=train_input, simulator=simulator)

    output_file.create_dataset('/seed%d/test_results' % randseed,
                               data=run_reservoir(period, 0, n_test, target_values, attractor_input=test_input,
                                                  test_recordings=test_recordings, simulator=simulator)[1])

    input_file.close()
    output_file.close()


def record_constant_rate_updates(target_values, update_prob=1.0, attractor_input=None,
                                 learning_rule='hebb', max_periods=400, filename='stochastic_updates.hdf',
                                 randseed=42, hebb_rate=0.0005, delayed_updates=False, hdf_folder=None):
    """
    Records an experiment with probabilistic updates for Hebbian learning (w/ and w/o the attractor).
    The network's state is memorised after each train period to be recovered after test.
    :param target_values:       a 1d signal to learn
    :param update_prob:         probability of an update
    :param attractor_input:     input from the attractor layer
    :param learning_rule:       hebb or force
    :param max_periods:         the maximum number of periods to compute for
    :param filename:            where to save the results
    :param randseed:            random seed of the reservoir
    :param hebb_rate:           learning rate (no decay, so it will be constant)
    :param delayed_updates:     if true, weights are updated in the end of each period
    :param hdf_folder:          float, value for a folder in the hdf file. None -> update_prob
    :return: None
    """
    period = len(target_values)
    reservoir = setup_reservoir(learning_rule=learning_rule, randseed=randseed,
                                hebb_rate=hebb_rate, hebb_const=True, update_prob=update_prob,
                                delayed_updates=delayed_updates)
    simulator = ReservoirInterface(reservoir, period)
    correlation = np.zeros(max_periods)
    updates = np.zeros(max_periods)

    for period in range(max_periods):
        test_results, curr_n_updates = run_reservoir(period, 1, 1, target_values, attractor_input=attractor_input,
                                                     test_recordings=['zhat'], simulator=simulator,
                                                     count_updates=True, freeze_train_activity=True)[1:]
        updates[period] = curr_n_updates
        correlation[period] = compute_shifted_cross_correlation(test_results, target_values)

    if hdf_folder is None:
        hdf_folder = update_prob

    output_file = h5py.File(filename, 'a')
    output_file.create_dataset('%.1f/seed%d/targets' % (hdf_folder, randseed), data=target_values)
    output_file.create_dataset('%.1f/seed%d/max_periods' % (hdf_folder, randseed), data=max_periods)
    output_file.create_dataset('%.1f/seed%d/hebb_rate' % (hdf_folder, randseed), data=hebb_rate)

    output_file.create_dataset('%.1f/seed%d/updates' % (hdf_folder, randseed), data=updates)
    output_file.create_dataset('%.1f/seed%d/correlation' % (hdf_folder, randseed), data=correlation)
    output_file.close()
