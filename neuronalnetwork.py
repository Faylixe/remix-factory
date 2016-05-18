#!/usr/bin/python

import logging
import pickle

import numpy

from os import makedirs
from os.path import join, exists
from multiprocessing import Pool

import configuration

#NEURON_SIZE = 0
#def createNeuron(i):
#    """ Static factory that allows Neuron to be created through multiprocessing. """
#    return Neuron(NEURON_SIZE, i)

class Neuron:
    """ Class that represents a neuron of our network. """

    def __init__(self, size, id, value=1):
        """ Default constructor. Initializes weight list with the given value. """
        self.file = join(configuration.NEURONS_DIRECTORY, configuration.NEURONS_FILE_PREFIX + str(id))
        weights = numpy.empty(size)
        # TODO : Add activation weight ?
        weights.fill(value)
        numpy.save(self.file, weights)

    def apply(self, vector):
        """ Applies neuronal computation to the given vector. """
        weights = numpy.load(self.file)
        result = 0 # TODO : Add activation weight ?
        for i in xrange(len(vector)):
            result = result + (vector[i] * weights[i])
        return result

    def train(self, sample, alpha):
        """ Train this neuron using the given sample and alpha learning coefficient. """
        output = self.apply(sample[0])
        weights = numpy.load(self.file)
        for i in xrange(len(weights)):
            weights[i] = weights[i] + (alpha * (sample[1][i] - output) * sample[0][i])
        numpy.save(self.file, weights)

class NeuronFactory:
    """ A NeuronFactory is in charge of creating Neuron through a pool. """

    def __init__(self, size):
        """ Default constructor. Initializes neuron size. """
        self.size = size

    def __call__(self, id):
        """ Factory method that creates a neuron of the target size with the given id."""
        return Neuron(self.size, id)

class NeuronalNetwork:
    """ Class that represents our neuronal network. """

    def __init__(self, size):
        """ Creates a untrained neuronal network with n neurons. """
        if not exists(configuration.NEURONS_DIRECTORY):
            makedirs(configuration.NEURONS_DIRECTORY)
        #NEURON_SIZE = size
        factory = NeuronFactory(size)
        pool = Pool(configuration.THREAD)
        self.neurons = pool.map(factory, xrange(size))
        pool.close()
        pool.join()

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
        size = len(self.neurons)
        result = numpy.empty(size)
        for i in xrange(n):
            result[i] = self.neurons[i].apply(vector)
        return result

    def load(path):
        """ Loads the neuronal network from the file denoted by the given path. """
        stream = open(path, 'rb')
        return pickle.load(stream)
