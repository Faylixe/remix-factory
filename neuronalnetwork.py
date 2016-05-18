#!/usr/bin/python

import logging
import pickle

import numpy

from os import makedirs
from os.path import join, exists
from multiprocessing import Pool

import configuration

NEURON_SIZE = 0

def createNeuron(i):
    """ """
    return Neuron(NEURON_SIZE, i)

class Neuron:
    """ Class that represents a neuron of our network. """

    def __init__(self, n, id, value=1):
        """ Default constructor. Initializes weight list with the given value. """
        self.file = join(configuration.NEURONS_DIRECTORY, configuration.NEURONS_FILE_PREFIX + str(id))
        weights = numpy.empty(n)
        weights.fill(value)
        self.save(weights)
        # TODO : Add activation weight ?

    def load(self):
        """ """
        stream = open(self.file, 'rb')
        return pickle.load(stream)

    def save(self, weights):
        """ """
        stream = open(self.file, 'wb')
        pickle.dump(weights, stream, -1)

    def apply(self, vector):
        """ Applies neuronal computation to the given vector. """
        weights = self.load()
        result = 0 # TODO : Add activation weight ?
        for i in xrange(len(vector)):
            result = result + (vector[i] * weights[i])
        return result

    def train(self, sample, alpha):
        """ Train this neuron using the given sample and alpha learning coefficient. """
        output = self.apply(sample[0])
        weights = self.load()
        for i in xrange(len(weights)):
            weights[i] = weights[i] + (alpha * (sample[1][i] - output) * sample[0][i])
        self.save(weights)

class NeuronalNetwork:
    """ Class that represents our neuronal network. """

    def __init__(self, n):
        """ Creates a untrained neuronal network with n neurons. """
        if not exists(configuration.NEURONS_DIRECTORY):
            makedirs(configuration.NEURONS_DIRECTORY)
        NEURON_SIZE = n
        pool = Pool(configuration.THREAD)
        self.neurons = pool.map(createNeuron, xrange(n))
        pool.close()
        pool.join()
        #for i in xrange(n):
        #    self.neurons.append(Neuron(n, i))

    def train(self, corpus, alpha):
        """ Trains this network using gradient descent. """
        i = 0
        for neuron in self.neurons:
            logging.info("Training neuron #%d" % i)
            i = i + 1
            for sample in corpus:
                neuron.train(sample, alpha)

    def save(self, path):
        """ Saves this neuronal network to the file denoted by the given path. """
        stream = open(path, 'wb')
        pickle.dump(self, stream, -1)

    def apply(self, vector):
        """ Transforms the given vector by applying each neuron to it. """
        n = len(self.neurons)
        result = numpy.empty(n)
        for i in xrange(n):
            result[i] = self.neurons[i].apply(vector)
        return result

    def load(path):
        """ Loads the neuronal network from the file denoted by the given path. """
        stream = open(path, 'rb')
        return pickle.load(stream)
