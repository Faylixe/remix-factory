#!/usr/bin/python

import logging
import numpy
import pickle

from backports import lzma
from os import makedirs, remove
from os.path import join, exists
from multiprocessing import Pool

import configuration

class Neuron:
    """ Class that represents a neuron of our network. """

    def __init__(self, size, id, value=1):
        """ Default constructor. Initializes weight list with the given value. """
        self.file = join(configuration.NEURONS_DIRECTORY, configuration.NEURONS_FILE_PREFIX + str(id))
        self.compressedFile = self.file + configuration.COMPRESSION_EXTENSION
        # TODO : Add activation weight ?
        pattern = numpy.array([value, value], dtype='int16')
        weights = numpy.tile(pattern, [size, 1])
        self.save(weights)

    def apply(self, vector):
        """ Applies neuronal computation to the given vector. """
        weights = self.load()
        result = [0, 0] # TODO : Add activation weight ?
        for i in xrange(len(vector)):
            result[0] = result[0] + (vector[i][0] * weights[i][0])
            result[1] = result[1] + (vector[i][1] * weights[i][1])
        return result

    def train(self, sample, alpha):
        """ Train this neuron using the given sample and alpha learning coefficient. """
        output = self.apply(sample[0])
        weights = self.load()
        for i in xrange(len(weights)):
            weights[i][0] = weights[i][0] + (alpha * (sample[1][i][0] - output) * sample[0][i][0])
            weights[i][1] = weights[i][1] + (alpha * (sample[1][i][1] - output) * sample[0][i][1])
        self.save()

    def save(self, weights):
        """ """
        weights.tofile(self.file)
        with open(self.file, "rb") as source:
            with lzma.open(self.compressedFile, "w") as compressor:
                compressor.write(source.read())
        remove(self.file)

    def load(self):
        """ """
        with open(self.file, "wb") as target:
            with lzma.open(self.compressedFile, "r") as uncompressor:
                target.write(uncompressor.read())
        weights = numpy.load(self.file)
        remove(self.file)
        return weights

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
        result = numpy.empty(size, dtype='int16')
        for i in xrange(n):
            result[i] = self.neurons[i].apply(vector)
        return result

    def load(path):
        """ Loads the neuronal network from the file denoted by the given path. """
        stream = open(path, 'rb')
        return pickle.load(stream)
