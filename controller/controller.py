#!/usr/bin/python

import sys
from multiprocessing import Manager, Pool

class Controller:
    """Base class for neuron Controller

    Such controller aims to perform operation over a set of neuron
    that are sent through a Pool operation. It holds an internal atomic
    counter and display progression over the given stream or standard input.
    """

    """ Shared manager instance used for creating lock and value. """
    manager = Manager()

    def __init__(self, limit, label, stream=sys.stdout):
        """Default constructor.

        :param limit: Number of neuron that will be processed by this controller.
        :param label: Output label to be displayed with progression.
        :param stream: (Optional) target stream in which progression will be written.
        """
        self.counter = Controller.manager.Value('i', 0)
        self.limit = limit
        self.label = label
        self.stream = stream

    def next(self):
        """ Increments counter and display progression. """
        self.counter.value += 1
        progress = '%s (%d/%d)' % (self.label, self.counter.value, self.limit)
        if self.counter.value < self.limit:
            progress += '\r'
        self.stream.write(progress)
        self.stream.flush()

    @staticmethod
    def run(controller, iterable, thread=None):
        """Runs the given controller using multiprocessing.

        :param controller: Controller instance to run.
        :param iterable: Input to submit to the controller.
        :param thread: (Optional) number of thread to use.
        """
        pool = Pool(thread)
        pool.apply_async(controller, iterable)
        pool.close()
        pool.join()
