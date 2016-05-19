#!/usr/bin/python

import logging
import numpy
import pickle
import time
import sys

from backports import lzma
from multiprocessing import Pool, Manager, Lock, Value
from os import makedirs, remove
from os.path import join, exists
from shutil import copyfile

import configuration

class Neuron:
    """ Class that represents a neuron of our network. """

    def __init__(self, directory, size, id):
        """ Default constructor. """
        self.size = size
        self.file = join(directory, configuration.NEURONS_FILE_PREFIX + str(id))
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

    def train(self, vector, learningRate):
        """ Train this neuron using the given sample and alpha learning coefficient. """
        output = self.apply(vector[0])
        weights = self.load()
        for i in xrange(len(weights)):
            for j in (0, 1):
                weights[i][j] = weights[i][j] + (learningRate * (vector[1][i][j] - output) * vector[0][i][j])
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

class NeuronMonitor:
    """ Base class for shared neuron processor. """

    def __init__(self, limit, label, manager):
        """ Default constructor. Initializes counter. """
        self.counter = manager.Value('i', 0)
        self.limit = limit
        self.label = label

    def next(self):
        """ Increments counter and display progression. """
        self.counter.value += 1
        progress = '%s (%d/%d)\r' % (self.label, self.counter.value, self.limit)
        sys.stdout.write(progress)
        sys.stdout.flush()


class NeuronFactory(NeuronMonitor):
    """ A NeuronFactory is in charge of creating Neuron through a pool. """

    def __init__(self, directory, size, manager, source, limit):
        """ Default constructor. Initializes factory attributes. """
        NeuronMonitor.__init__(self, limit, "Creating neuron", manager)
        self.size = size
        self.lock = manager.Lock()
        self.source = source
        self.directory = directory

    def __call__(self, id):
        """ Factory method that creates a neuron of the target size with the given id."""
        neuron = Neuron(self.directory, self.size, id)
        copyfile(self.source.getCompressedFile() , neuron.getCompressedFile())
        with self.lock:
            self.next()
            time.sleep(0.001)
        return neuron

class NeuronTrainer(NeuronMonitor):
    """ A NeuronTrainer is in charge of training a given neuron through a pool. """

    def __init__(self, corpus, size, window, manager, learningRate):
        """ Default constructor. Initializes trainer attributes. """
        NeuronMonitor.__init__(self, size, "Training neuron", manager)
        self.lock = manager.Lock()
        self.learningRate = learningRate
        self.corpus = corpus
        self.size = size
        self.window = window

    def __call__(self, neuron):
        """ Training method that applies the given corpus to the given neuron. """
        for vector in self.corpus:
            if self.size == self.window:
                neuron.train(vector, self.learningRate)
            else:
                subvector = None
                if i < self.window:
                    subvector = vector[0:self.window]
                elif i > self.size - self.window:
                    subvector = vector[self.size - self.window:]
                else:
                    subvector = vector[i:i + self.window]
                neuron.train(subvector, self.learningRate)
        with self.lock:
            self.next()
            time.sleep(0.001)

class NeuronalNetwork:
    """ Class that represents our neuronal network. """

    def __init__(self, size, directory, window, thread):
        """ Creates a untrained neuronal network with n neurons. """
        self.directory = directory
        self.pool = Pool(thread)
        self.manager = Manager()
        self.size = size
        self.window = window

    def create(self):
        """ Initializes this neuronal network. """
        if not exists(self.directory):
            makedirs(self.directory)
        source = Neuron(self.directory, self.window, 'source')
        source.reset()
        factory = NeuronFactory(self.directory, self.window, self.manager, source, self.size)
        self.neurons = self.pool.map(factory, xrange(self.size))
        self.pool.close()
        self.pool.join()
        with open(join(self.directory, configuration.MODEL_METADATA), 'w') as metadata:
            metadata.write(self.size)
        print ''

    def train(self, corpus, learningRate):
        """ Trains this network using gradient descent. """
        if not exists(self.directory):
            raise IOError('Directory %s not found, abort' % self.directory)
        trainer = NeuronTrainer(corpus, self.size, self.window, self.manager, learningRate)
        self.pool.apply_async(trainer, self.neurons)
        self.pool.close()
        self.pool.join()

    def save(self, path):
        """ Saves this neuronal network to the file denoted by the given path. """
        stream = open(path, 'wb')
        pickle.dump(self, stream, -1)

    def apply(self, vector):
        """ Transforms the given vector by applying each neuron to it. """
        size = len(self.neurons)
        monitor = Bar('Creating remix', max=len(size))
        result = numpy.empty(size, dtype='int16')
        for i in xrange(size):
            if self.size == self.window:
                result[i] = self.neurons[i].apply(vector)
            else:
                subvector = None
                if i < self.window:
                    subvector = vector[0:self.window]
                elif i > self.size - self.window:
                    subvector = vector[self.size - self.window:]
                else:
                    subvector = vector[i:i + self.window]
                result[i] = self.neurons[i].apply(subvector)
            monitor.next()
        return result

    def load(path):
        """ Loads the neuronal network from the file denoted by the given path. """
        stream = open(path, 'rb')
        return pickle.load(stream)

    def getSize(directory):
        """ Retrieves and returns size of model denoted by the given directory. """
        with open(join(directory, configuration.MODEL_METADATA), 'r') as file:
            return int(next(file))
