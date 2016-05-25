
#!/usr/bin/python

import time

from controller import Controller

class NeuronFactory(Controller):
    """A NeuronTrainer is in charge of training a given neuron.

    It uses a source neuron as a template for all other neurons to be created,
    and copy this source neuron each time the __call__ method is used.
    """

    def __init__(self, source, size):
        """Default constructor.

        :param source: Path of the source file the neuron will be created from.
        :param size: Number of neuron to be created through this controller.
        """
        Controller.__init__(self, size, "Creating neurons")
        self.source = source

    def __call__(self, neuron):
        """Creates empty model as compressed file for the given neuron.

        :param neuron: Neuron to create file for.
        """
        copyfile(self.source, neuron.getCompressedFile())
        self.next()
