#!/usr/bin/python

import logging
import pickle

LOGGER = logging.getLogger('neuronal-network')

class Neuron:
    """ Class that represents a neuron of our network. """

    def __init__(self, n, value=1):
        """ Default constructor. Initializes weight list with the given value. """
        self.weights = [value for i in xrange(n)]
        # TODO : Add activation weight ?

    def apply(self, vector):
        """ Applies neuronal computation to the given vector. """
        result = 0 # TODO : Add activation weight ?
        for i in xrange(len(vector)):
            result = result + (vector[i] * weights[i])
        return result

    def train(self, sample, alpha):
        """ Train this neuron using the given sample and alpha learning coefficient. """
        output = self.apply(sample[0])
        for i in xrange(len(self.weights)):
            self.weights[i] = self.weights[i] + (alpha * (sample[1][i] - output) * sample[0][i])


class NeuronalNetwork:
    """ Class that represents our neuronal network. """

    def __init__(self, n):
        """ Creates a untrained neuronal network with n neurons. """
        self.neurons = []
        for i in xrange(n):
            self.neurons.append(Neuron())

    def train(self, corpus, alpha):
        """ Trains this network using gradient descent. """
        i = 0
        for neuron in self.neurons:
            LOGGER.info("Training neuron #%d" % i)
            i = i + 1
            for sample in corpus:
                neuron.train(sample, alpha)

    def save(self, path):
        """ Saves this neuronal network to the file denoted by the given path. """
        stream = open(path, 'wb')
        pickle.dump(self, stream, -1)

    def apply(self, vector):
        """ Transforms the given vector by applying each neuron to it. """
        result = []
        for neuron in self.neurons:
            result.append(neuron.apply(vector))
        return result

    def load(path):
        """ Loads the neuronal network from the file denoted by the given path. """
        stream = open(path, 'rb')
        return pickle.load(stream)
