#!/usr/bin/python

import numpy

from backports import lzma
from os import makedirs, remove
from os.path import join, exists

from model import NEURON_EXTENSION, NEURON_PREFIX

class Neuron:
    """Class that represents a neuron of our network.

    Such neuron is defined as a path to a persistent file which contains our
    neuron's values. Such file are known as CBN (Compressed Binary Neuron) which
    consists in a serialized numpy vector compressed using LZMA.

    All operations on this neurons will performs a loading then a saving of it
    internal state.
    """

    def __init__(self, directory, size, id):
        """ Default constructor.

        :param directory: Path of the directory in which this neuron will be stored.
        :param size: Size of this neuron, which correspond to the weight vector length.
        :param id: Identifier of this neuron.
        """
        if not exists(directory):
            makedirs(directory)
        self.size = size
        self.file = join(directory, NEURON_PREFIX + str(id))
        self.compressedFile = self.file + NEURON_EXTENSION

    def getCompressedFile(self):
        """Getter for compressed file path.

        :returns: Path of the neuron compressed file.
        """
        return self.compressedFile

    def reset(self, value=1):
        """Initializes weight list with the given value.

        Creates a numpy array as weight vector, filled with value.
        The created weight vector is then saved to the target
        compressed file.
        :param value: Default value to use as weight.
        """
        pattern = numpy.array([value, value], dtype='float64')
        weights = numpy.tile(pattern, [self.size, 1])
        self.save(weights)

    def apply(self, vector):
        """Applies neuronal computation to the given vector.

        The neuron's weights are applied to the given vector
        as following :

        :math:`\\sum\\limits_{i=0}^n w_{i0} * x_{i0}`,

        for the left chanel and

        :math:`\\sum\\limits_{i=1}^n w_{i0} * x_{i1}`,

        for the right one.

        :param vector: Vector to apply neuron to. Such vector should be a numpy array of [left, right] values.
        :returns: Transformed vector.
        """
        weights = self.load()
        result = [0, 0]
        for i in xrange(len(vector)):
            for j in (0, 1):
                result[j] = result[j] + (vector[i][j] * weights[i][j])
        return result

    def train(self, vector, learningRate):
        """Train this neuron using the given sample and alpha learning coefficient.

        Training use the gradient descent algorithm to improve internal neuron's
        weights as following :

        :math:`w_{ij} = w_{ij} + \\alpha * \\x_{ij} * (y_{ij} - f(x_{ij}))`,

        where :math:`f(x_{ij}) = \\sum\\limits_{i=0}^n w_{ij} * x_{ij}`

        :param vector: Vector to use for training.
        :param learningRate: Learning rate (alpha parameter) used for training.
        """
        output = self.apply(vector[0])
        weights = self.load()
        for i in xrange(max(len(original), len(remixed))):
            for j in (0, 1):
                originalValue = [0, 0]
                remixedValue = [0, 0]
                if len(vector[0]) < i:
                    originalValue = vector[0][i]
                if len(vector[1]) < i:
                    remixedValue = vector[1][i]
                weights[i][j] = weights[i][j] + (learningRate * (remixedValue[j] - output) * originalValue[j])
        self.save(weights)

    def save(self, weights):
        """Saves the given weights to this neuron associated file.

        Weights vector is first saved as serialized numpy array, then LZMA
        compression algorithm is applied in order to minimize neuron file weight.

        :param weights: Weights vector to save.
        """
        weights.tofile(self.file)
        with open(self.file, "rb") as source:
            with lzma.open(self.compressedFile, "w") as compressor:
                compressor.write(source.read())
        remove(self.file)

    def load(self):
        """Loads the neuron weights from the associated file and returns it.

        :returns: Loaded weights vector.

        """
        with open(self.file, "wb") as target:
            with lzma.open(self.compressedFile, "r") as uncompressor:
                target.write(uncompressor.read())
        weights = numpy.load(self.file)
        remove(self.file)
        return weights
