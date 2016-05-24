#!/usr/bin/python

import numpy

from os.path import exists

from controller.train import NeuronTrainer
from controller.create import NeuronFactory
from.controller.controller import Controller

class NeuronalNetwork:
    """Class that represents our neuronal network.

    """

    def __init__(self, directory, size, window):
        """Creates a untrained neuronal network with size neurons.

        :param directory:
        :param size:
        :param window:
        """
        self.directory = directory
        self.size = size
        self.window = window
        self.neurons = Neurons(directory, window, size)

    def create(self, thread=None):
        """Creates each neuron with default valued weights.

        It first creates a original neuron that will act as template. Then it
        will copy this file in parallel, provisioning the internal neurons
        reference collection at the same time.

        :param thread: (Optional) Number of thread to use for creating neuron file.
        """
        source = Neuron(self.directory, self.window, 'origin')
        source.reset()
        factory = NeuronFactory(source.getCompressedFile(), self.size)
        controller.run(factory, self.neurons, thread)

    def train(self, corpus, learningRate, thread=None):
        """ Trains this network using gradient descent.

        :param corpus:
        :param learningRate: Learning rate (alpha parameter) used for training.
        :param thread: (Optional) Number of thread to use for training neurons.
        """
        if not exists(self.directory):
            raise IOError('Directory %s not found, abort' % self.directory)
        trainer = NeuronTrainer(corpus, self.size, self.window, learningRate)
        controller.run(trainer, self.neurons, thread)

    def apply(self, vector):
        """Transforms the given vector by applying each neuron to it.

        :param vector:
        ADD ERROR here.
        """
        if (self.size < len(vector)):
            raise IOError('Given vector is too big for this neuronal network')
        result = numpy.empty(self.size, dtype='int16')
        for i in xrange(self.size):
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

    def save(self, path):
        """Saves this neuronal network to the file denoted by the given path.

        The network is saved as a JSON file containing only model metadata including :

        * Model directory
        * Model size
        * Neuron window value

        :param path: Path of the file to write.
        """
        metadata = {'window': self.window, 'size': self.size, 'directory': self.directory}
        with open(path, 'w') as stream:
            json.dump(metadata, stream)

    @staticmethod
    def load(path):
        """Loads the neuronal network from the file denoted by the given path.

        The network is loaded as a JSON file containing only model metadata, including :

        * Model directory
        * Model size
        * Neuron window value

        :param path: Path of the file to load as NeuronalNetwork instance.
        :returns: Loaded instance.
        """
        with open(path, 'r') as stream:
            metadata = json.loads(stream)
            size = metadata['size']
            window = metadata['window']
            directory = metadata['directory']
            network = NeuronalNetwork(directory, size, window)
