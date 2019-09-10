# -*- coding: utf-8 -*-

"""
    File name: attractor_net.py
    Description: a network of rate neurons with attractor dynamics
    Python version: 3.6
"""

import numpy as np
import math
from scipy.stats import norm as gaussian_distr
from warnings import warn
from utils import relu


class AttractorNetwork:
    """
    A rate network with Gaussian symmetric connectivity and the rectifier nonlinearity.

    Adapted from
    Itskov, V., Curto, C., Pastalkova, E., & Buzsaki, G. (2011)
    Cell Assembly Sequences Arising from Spike Threshold Adaptation
    Keep Track of Time in the Hippocampus.
    Journal of Neuroscience, 31(8), 2828–2834.
    https://doi.org/10.1523/JNEUROSCI.3773-10.2011
    """
    def __init__(self, weights_rec_filename=None, noisy_weights_std=1.0,
                 randseed=None, num_rec=2500):
        """
        :param weights_rec_filename: name of the file with recurrent weight or None
        :param noisy_weights_std:    standard deviation of noisy part of the weights
        :param randseed:             random seed for all random operations
        :param num_rec:              number of recurrent neurons
        """
        self._rand_generator = np.random.RandomState(randseed)

        self._num_rec = num_rec  # Network size
        self._net_side = int(math.sqrt(self._num_rec))

        if self._net_side != round(self._net_side):
            warn('number of neurons must be a perfect square')

        self._time_step = 1.0  # ms
        
        # firing rate related constants
        self._tau_membrane = 30.0  # ms
        self._tau_adaptation = 400.0  # ms
        self._adaptation_strength = 1.5  # rates multiplier in adaptation dynamics

        self._rates_decay = self._time_step / self._tau_membrane
        self._adaptation_decay = self._time_step / self._tau_adaptation

        # Connectivity constants
        self._weights_offset = -0.375  # symmetric weights offset
        self._weights_width = 1.0      # symmetric weights width
        self._noisy_weights_std = noisy_weights_std / math.sqrt(self._num_rec)  # noisy weights width

        # Random Gaussian input constants
        self._input_noise_mean = 1.0
        self._input_noise_std = 0.0025

        self.firing_rates = np.zeros(self._num_rec)
        self.adaptation = np.zeros(self._num_rec)

        self.square_rates = np.zeros((self._net_side, self._net_side))
        self.position = np.zeros(2)
        self.inputs = np.zeros(self._num_rec)

        if weights_rec_filename is None:
            self._make_rec_weights()
        else:
            self.weights_rec = np.load(weights_rec_filename)

        self._direction_vector = np.exp(2.j * np.pi * np.linspace(0, self._net_side - 1, self._net_side) /
                                        self._net_side)

    def _compute_symmetric_weights(self):
        """Computes Gaussian symmetric connectivty with periodic boundaries."""
        xis = np.repeat(np.arange(self._net_side), self._num_rec * self._net_side)
        yis = np.tile(np.repeat(np.arange(self._net_side), self._num_rec), self._net_side)
        xjs = np.tile(np.repeat(np.arange(self._net_side), self._net_side), self._num_rec)
        yjs = np.tile(np.arange(self._net_side), self._net_side * self._num_rec)

        distance_x = np.minimum(abs(xis - xjs), self._net_side - abs(xis - xjs)) * math.pi / self._net_side
        distance_y = np.minimum(abs(yis - yjs), self._net_side - abs(yis - yjs)) * math.pi / self._net_side
        distance = np.sqrt(distance_x ** 2 + distance_y ** 2)

        return self._weights_offset + gaussian_distr.pdf(distance, 0.0, self._weights_width).reshape(
            (self._num_rec, self._num_rec))

    def _make_rec_weights(self):
        """Makes symmetric and noisy weights."""
        self.weights_rec = self._compute_symmetric_weights()
        self.weights_rec += self._rand_generator.normal(0, self._noisy_weights_std, (self._num_rec, self._num_rec))

    def _update_inputs(self, external_input=None):
        """Updates noisy and external input , if it is given."""
        self.inputs = self._rand_generator.normal(self._input_noise_mean, self._input_noise_std, self._num_rec)
        if external_input is not None:
            self.inputs += external_input

    def _compute_position(self, mean_rates):
        """Computes centre of 1d rates, converting a line to a circle on a complex plane."""
        return (np.angle(np.dot(mean_rates, self._direction_vector)) / np.pi * self._net_side / 2.0) % self._net_side

    def _update_position(self):
        """Updates 2d position of the bump's centre."""
        self.position[0] = self._compute_position(self.square_rates.mean(0))
        self.position[1] = self._compute_position(self.square_rates.mean(1))

    def _update_firing_rates(self):
        """Updates rate's dynamics and position."""
        self.firing_rates += (self._rates_decay * (-self.firing_rates + relu(
            np.dot(self.weights_rec, self.firing_rates) + self.inputs - self.adaptation))).round(10)
        # round to reduce errors
        
        self.square_rates = self.firing_rates.reshape((self._net_side, self._net_side), order='F')
        self._update_position()

    def _update_adaptation(self):
        """Updates adaptation's dynamics."""
        self.adaptation += self._adaptation_decay * (- self.adaptation + self._adaptation_strength * self.firing_rates)

    def update(self, external_input=None):
        """Updates network's dynamics."""
        self._update_inputs(external_input)
        self._update_firing_rates()
        self._update_adaptation()

    def save_weights(self, filename):
        """Saves recurrent weights."""
        np.save(filename, self.weights_rec)

    def reset_network(self):
        """Resets network's state to zero."""
        self.firing_rates.fill(0.0)
        self.adaptation.fill(0.0)
        self.square_rates.fill(0.0)
        self.position.fill(0.0)
        self.inputs.fill(0.0)
