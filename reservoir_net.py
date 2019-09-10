# -*- coding: utf-8 -*-

"""
    File name: reservoir_net.py
    Description: a neural network with readout units and a feedback loop
    Python version: 3.6
"""

import numpy as np
from warnings import warn


class ReservoirNet(object):
    """
    A recurrent neural net with trainable output units. Variable names are chosen to be consistent
    with the reference paper.

    This is an implementation of
    Hoerzer, G. M.; Legenstein, R. & Maass, W.
    Emergence of Complex Computational Structures From Chaotic Neural Networks
    Through Reward-Modulated Hebbian Learning. Cereb Cortex, 2012
    """
    def __init__(self, num_rec=1000, num_out=1, amplitude_fb_weights=1.0, randseed=None,
                 learning_rule='hebb', hebb_rate=0.0005, hebb_decay=20.0 * 1e3, hebb_const=False, update_prob=1.0,
                 delayed_updates=False):
        """
        :param num_rec:                 number of recurrent neurons
        :param num_out:                 number of output units (== target function dimension)
        :param amplitude_fb_weights:    strength of feedback connection from the output units
        :param randseed:                random seed for all random operations
        :param learning_rule:           'hebb' for reward-modulated Hebbian rule of 'force' for FORCE
        :param hebb_rate:               initial learning rate for the Hebbian rule
        :param hebb_decay:              hyperbolic decay constant for the Hebbian rule's learning rate
        :param hebb_const:              whether Hebbian learning uses constant learning rate or not
        :param update_prob:             probability of the weights' update
        :param delayed_updates:         if true, all updates are collected and applied on apply_weight_updates()
        """
        self._num_rec = num_rec
        self._num_out = num_out
        self._theta_state = 0.05  # amplitude of firing rates noise
        self._theta_expl = 0.5  # amplitude of output units noise
        self._rec_connect_prob = 0.1  # probability of a recurrent connection
        self._tau = 50.0  # membrane time constant, ms
        self._tau_avg = 5.0  # decay constant for smoothing filters
        self._dt = 1.0  # time step, ms
        self._lambda = 1.5  # strength of recurrent connection, controls the chaotic regime
        self._eta_init = hebb_rate  # initial learning rate
        self._tau_eta = hebb_decay  # decay constant for the learning rate, ms
        self._hebb_const = hebb_const  # constant learning rate; makes hebb_decay useless
        self._update_prob = update_prob

        self._rand_generator = np.random.RandomState(randseed)
        self._amplitude_fb_weights = amplitude_fb_weights
        self._external_noise_amplitude = 0.05  # Gaussian noise amplitude for input signals

        self.x = np.zeros(self._num_rec)  # membrane potential
        self.r = np.zeros(self._num_rec)  # firing rate
        self.z = np.zeros(self._num_out)  # noisy output
        self.zhat = np.zeros(self._num_out)  # noiseless output
        self.zbar = np.zeros(self._num_out)  # temporally smoothed noisy output
        self.f = np.zeros(self._num_out)  # target function
        self.weights_out = self._make_weights_out()  # output weights (plastic)
        
        self._xi_state = np.zeros(self._num_rec)  # firing rates noise
        self._xi_expl = np.zeros(self._num_out)  # output noise
        self._weights_rec = self._make_weights_rec()  # recurrent weights
        self._weights_fb = self._make_weights_fb()  # feedback weights
        self._filter_decay = self._dt / self._tau_avg
        self._dynamics_decay = self._dt / self._tau
        self._external_input = np.zeros(self._num_rec)
        self._time = 0  # inner time, used for the learning rates
        self._learning_rule = learning_rule

        self._update_indicator = 1.0  # for stochastic weights update

        if self._learning_rule == 'hebb':
            self._eta = self._eta_init
            self._p = 0.
            self._pbar = 0.
            self._modulation = 0.
        elif self._learning_rule == 'force':
            self._covariance = np.eye(self._num_rec)  # times alpha, which is 1.0 in the FORCE paper
            self._cov_r_prod = self._covariance.dot(self.r)

        self._memorised_state = None  # used to memorise the state after training

        self._delayed_updates = delayed_updates
        if self._delayed_updates:
            self._collected_updates = np.zeros_like(self.weights_out)

    def _make_indicator_rec(self):
        """
        Creates a binary recurrent connectivity matrix.
        :return: a (_num_rec, _num_rec) 0/1 matrix
        """
        size = (self._num_rec, self._num_rec)
        return self._rand_generator.binomial(1, self._rec_connect_prob, size=size)

    def _make_weights_rec(self):
        """
        Makes random normal recurrent weights.
        :return: a (_num_rec, _num_rec) real matrix
        """
        size = (self._num_rec, self._num_rec)
        std = np.sqrt(1.0 / (self._rec_connect_prob * self._num_rec))
        return self._rand_generator.normal(loc=0.0, scale=std, size=size) * self._make_indicator_rec()

    def _make_weights_fb(self):
        """
        Makes random uniform feedback weights.
        :return: a (_num_rec, _num_out) real matrix
        """
        return self._rand_generator.uniform(low=-self._amplitude_fb_weights,
                                            high=self._amplitude_fb_weights,
                                            size=(self._num_rec, self._num_out))

    def _make_weights_out(self):
        """
        Sets initial output weights to zero.
        :return: a zero (_num_out, _num_rec) matrix
        """
        return np.zeros((self._num_out, self._num_rec))

    def _get_dx(self):
        """Return update of the membrane potential."""
        return self._dynamics_decay * (-self.x + self._lambda * np.dot(self._weights_rec, self.r) +
                                       np.dot(self._weights_fb, self.z) + self._external_input)

    def _set_xi_state(self):
        """Sets noise in firing rates"""
        self._xi_state = self._rand_generator.uniform(low=-self._theta_state,
                                                      high=self._theta_state, size=self._num_rec)

    def _set_r(self):
        """Sets firing rates (including noise)"""
        self.r = np.tanh(self.x) + self._xi_state

    def _set_xi_expl(self):
        """Sets noise in readout neurons"""
        self._xi_expl = self._rand_generator.uniform(low=-self._theta_expl,
                                                     high=self._theta_expl, size=self._num_out)

    def _set_zhat(self):
        """Sets activity in readout neurons (without noise)"""
        self.zhat = np.dot(self.weights_out, self.r)

    def _set_z(self):
        """Sets activity in readout neurons (including noise)"""
        self.z = self.zhat + self._xi_expl

    def _set_zbar(self):
        """Sets low-pass filtered version of activity in readout neurons (including noise)"""
        self.zbar = (1.0 - self._filter_decay) * self.zbar + self._filter_decay * self.z

    def _update_external_input(self, external_input=None):
        """Updates external input and adds noise (if None, sets it to zero)."""
        if external_input is not None:  # add noise
            self._external_input = external_input + self._external_noise_amplitude * \
                                   self._rand_generator.normal(loc=0.0, scale=1.0, size=(self._num_rec,))
        else:
            self._external_input.fill(0)

    def _update_hebb_params(self):
        """Updates parameters of the reward-modulated Hebbian learning rule."""
        self._p = -np.dot(self.z - self.f, self.z - self.f)
        self._pbar = (1.0 - self._filter_decay) * self._pbar + self._filter_decay * self._p
        self._modulation = float(self._p > self._pbar)
        if not self._hebb_const:
            self._eta = self._eta_init / (1.0 + self._time / self._tau_eta)

    def _update_force_params(self):
        """Updates _covariance estimate of the FORCE rule."""
        self._cov_r_prod = self._covariance.dot(self.r)  # old cov, new r
        self._covariance -= np.outer(self._cov_r_prod, self._cov_r_prod / (1.0 + self.r.dot(self._cov_r_prod)))

    def _get_dw_out(self):
        """Returns learning rule-dependent weights update."""
        if self._learning_rule == 'hebb':
            return np.outer(self._modulation * self._eta * (self.z - self.zbar), self.r)
        if self._learning_rule == 'force':
            self._cov_r_prod = self._covariance.dot(self.r)  # new cov, new r
            return np.outer(self.f - self.zhat, self._cov_r_prod)

    def _update_activity(self):
        """Updates activity of the network."""
        self.x += self._get_dx()
        self._set_xi_state()
        self._set_r()
        self._set_xi_expl()
        self._set_zhat()  # noiseless readout
        self._set_z()    # noisy readout
        self._set_zbar()  # average

    def _update_plasticity(self, target_function):
        """
        Updates plasticity of the network.

        Must be called after _update_activity().
        :param target_function: a (_num_out) array (or a scalar for _num_out=1)
        :return: None
        """
        self._time += 1  # only used for plasticity (self._eta decay)
        self.f = target_function

        if self._learning_rule == 'hebb':
            self._update_hebb_params()
        elif self._learning_rule == 'force':
            self._update_force_params()

        if self._update_prob < (1.0 - 1e-12):
            self._update_indicator = self._rand_generator.binomial(1, self._update_prob)

        if self._delayed_updates:
            self._collected_updates += self._get_dw_out() * self._update_indicator
        else:
            self.weights_out += self._get_dw_out() * self._update_indicator

    def reset_network_activity(self):
        """
        Sets dynamics of the network to zero.
        :return: None
        """
        self.x = np.zeros(self._num_rec)
        self.r = np.zeros(self._num_rec)
        self.z = np.zeros(self._num_out)
        self.zhat = np.zeros(self._num_out)
        self.zbar = np.zeros(self._num_out)

    def memorise_state(self):
        """Saves the network's state. Does not save plasticity parameters."""
        self._memorised_state = (self.x.copy(), self.r.copy(), self.z.copy(), self.zhat.copy(), self.zbar.copy())

    def recover_memorised_state(self):
        """Recovers the saved network's state."""
        if self._memorised_state is not None:
            self.x, self.r, self.z, self.zhat, self.zbar = self._memorised_state
            self._memorised_state = None
        else:
            warn("Nothing to recover")

    def update(self, external_input=None, target_function=None):
        """
        Updates the state of the network and the output weights (if target_function is not None).
        :param external_input:  None or a (_num_rec) vector. In the latter case Gaussian noise will be added.
        :param target_function: None (for no plasticity) or a (_num_out) array.
        :return: None
        """
        self._update_external_input(external_input)
        self._update_activity()

        if target_function is not None:
            self._update_plasticity(target_function)

    def are_weights_updated(self):
        """Indicates whether the weights were updated last time or not."""
        if self._learning_rule == 'hebb':
            return int(self._modulation * self._update_indicator)
        return int(self._update_indicator)

    def are_updates_delayed(self):
        return self._delayed_updates

    def apply_weight_updates(self):
        if not self._delayed_updates:
            warn('Updates were not collected')
            return

        self.weights_out += self._collected_updates
        self._collected_updates.fill(0.0)
