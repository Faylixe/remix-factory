#!/usr/bin/python

from model.neuron import Neuron

class Neurons:
    """Neurons is a Neuron cache iterator.

    When first used it creates neuron instance on the fly and put them in a
    local cache which is then used as iterable once all are loaded.
    """

    def __init__(self, directory, size, limit):
        """Default constructor.

        :param directory: Path of the directory in which target neurons will be loaded from.
        :param size: Size of loaded neuron, which correspond to the weight vector length.
        :param limit: Iteration limit.
        """
        self.directory = directory
        self.neurons = []
        self.size = size
        self.limit = limit
        self.current = 0

    def __iter__(self):
        """Iterator interface implementation.

        :returns: Reference to this instance or cache if all neurons are created.
        """
        if self.current == self.limit:
            return self.neurons.__iter__()
        return self

    def next(self):
        """Retrieves next neuron and returns it.

        Created neuron will be added to the local cache.

        :returns: Retrieves neuron instance.
        """
        if self.current >= self.limit:
            raise StopIteration()
        neuron = Neuron(self.directory, self.size, self.current)
        self.neurons.append(neuron)
        self.current += 1
        return neuron
