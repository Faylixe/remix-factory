#!/usr/bin/python

import logging
import numpy
import pickle
import time

from backports import lzma
from multiprocessing import Pool, Manager, Lock
from os import makedirs, remove
from os.path import join, exists
from progress.bar import Bar
from shutil import copyfile

import configuration
import console

class Neuron:
    """ Class that represents a neuron of our network. """

    def __init__(self, size, id):
        """ Default constructor. """
        self.size = size
        self.file = join(configuration.NEURONS_DIRECTORY, configuration.NEURONS_FILE_PREFIX + str(id))
        self.compressedFile = self.file + configuration.COMPRESSION_EXTENSION

    def getCompressedFile(self):
        """ Getter for compressed file path. """
        return self.compressedFile

    def reset(self, value=1):
        """ Initializes weight list with the given value. """
        pattern = numpy.array([value, value], dtype='int16')
        weights = numpy.tile(pattern, [self.size, 1])
        # TODO : Add activation weight ?
        self.save(weights)

    def apply(self, vector):
        """ Applies neuronal computation to the given vector. """
        weights = self.load()
        result = [0, 0] # TODO : Add activation weight ?
        for i in xrange(len(vector)):
            for j in (0, 1):
                result[j] = result[j] + (vector[i][j] * weights[i][j])
        return result

    def train(self, sample, alpha):
        """ Train this neuron using the given sample and alpha learning coefficient. """
        output = self.apply(sample[0])
        weights = self.load()
        for i in xrange(len(weights)):
            for j in (0, 1):
                weights[i][j] = weights[i][j] + (alpha * (sample[1][i][j] - output) * sample[0][i][j])
        self.save(weights)

    def save(self, weights):
        """ Saves the given weights to this neuron associated file. """
        weights.tofile(self.file)
        with open(self.file, "rb") as source:
            with lzma.open(self.compressedFile, "w") as compressor:
                compressor.write(source.read())
        remove(self.file)

    def load(self):
        """ Loads the neuron weights from the associated file and returns it. """
        with open(self.file, "wb") as target:
            with lzma.open(self.compressedFile, "r") as uncompressor:
                target.write(uncompressor.read())
        weights = numpy.load(self.file)
        remove(self.file)
        return weights

class NeuronFactory:
    """ A NeuronFactory is in charge of creating Neuron through a pool. """

    def __init__(self, size, lock, source, monitor):
        """ Default constructor. Initializes factory attributes. """
        self.size = size
        self.lock = lock
        self.source = source
        self.monitor = monitor

    def __call__(self, id):
        """ Factory method that creates a neuron of the target size with the given id."""
        neuron = Neuron(self.size, id)
        copyfile(self.source.getCompressedFile() , neuron.getCompressedFile())
        with self.lock:
            self.monitor.next()
            time.sleep(0.001)
        return neuron

class NeuronTrainer:
    """ A NeuronTrainer is in charge of training a given neuron through a pool. """

    def __init__(self, corpus, lock, monitor, alpha):
        """ Default constructor. Initializes trainer attributes. """
        self.lock = lock
        self.monitor = monitor
        self.alpha = alpha
        self.corpus

    def __call__(self, neuron):
        """ Training method that applies the given corpus to the given neuron. """
        for sample in self.corpus:
            neuron.train(sample, self.alpha)
        with self.lock:
            self.monitor.next()
            time.sleep(0.001)

class NeuronalNetwork:
    """ Class that represents our neuronal network. """

    def __init__(self, size):
        """ Creates a untrained neuronal network with n neurons. """
        if not exists(configuration.NEURONS_DIRECTORY):
            makedirs(configuration.NEURONS_DIRECTORY)
        pool = Pool(configuration.THREAD)
        manager = Manager()
        lock = manager.Lock()
        monitor = Bar('Creating neurons', max=size)
        source = Neuron(size, id)
        source.reset()
        monitor.next()
        factory = NeuronFactory(size, lock, source, monitor )
        self.neurons = pool.map(factory, xrange(size))
        pool.close()
        pool.join()

    def train(self, corpus, alpha):
        """ Trains this network using gradient descent. """
        pool = Pool(configuration.THREAD)
        manager = Manager()
        lock = manager.Lock()
        monitor = Bar('Training neurons', max=len(self.neurons))
        trainer = NeuronTrainer(lock, monitor, alpha, corpus)
        pool.apply_async(trainer, self.neurons)
        pool.close()
        pool.join()

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
