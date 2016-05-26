#!/usr/bin/python

import time

from controller import Controller

class NeuronTrainer(Controller):
    """A NeuronTrainer is in charge of training a given neuron.


    """

    def __init__(self, corpus, size, window, learningRate):
        """Default constructor.

        :param corpus:
        :param size:
        :param window:
        :param learningRate:
        """
        Controller.__init__(self, size, "Training neuron")
        self.learningRate = learningRate
        self.corpus = corpus
        self.size = size
        self.window = window
        self.lock = Controller.manager.Lock()

    def __call__(self, neuron):
        """Training method that applies the given corpus to the given neuron.

        :param neuron: Neuron to train using internal corpus.
        """
        for vector in self.corpus:
            if self.size == self.window:
                neuron.train(vector, self.learningRate)
            else:
                i = neuron.getIdentifier()
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
            #time.sleep(0.001) # TODO : Check if useful.
