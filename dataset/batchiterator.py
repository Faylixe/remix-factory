#!/usr/bin/python

import logging

from os import listdir, makedirs
from os.path import isfile, join, exists, splitext, basename
from scipy.io import wavfile

from dataset.converters import convert, supported

""" Name of the child directory which contains original songs. """
ORIGINAL = 'original'

""" Name of the child directory which contains remixed songs. """
REMIXED = 'remixed'

class BatchIterator:
    """Class that defines a corpus with batch facilities.

    TODO : Write documentation.
    """

    @staticmethod
    def getDirectory(root, child):
        """ Builds and returns path of the directory denoted by root+child.

        If such directory does not exist, it will raise an IOError.

        :param root: Path of the corpus root directory.
        :param child: Name of the child directory within root.
        :returns: Absolute path of the child directory if exist.
        """
        path = join(root, child)
        if not exists(path):
            raise IOError('Directory %s does not exist' % Path)
        return path

    @staticmethod
    def load(file):
        """Loads the given wave file numerical values.

        :param file: Wav file to load value from.
        :returns: Wav file numerical value as a numpy array.
        """
        if supported(file):
            file = convert(file)
        _, data = wavfile.read(file)
        return data

    def __init__(self, directory, batchSize):
        """Default constructor.

        Ensures given corpus directory exists and is valid (contains original
        and remixed child directory). Then indexes all original songs.

        :param directory: Directory to extract song from.
        :param batchSize:
        """
        self.originalDirectory = self.getDirectory(directory, ORIGINAL)
        self.remixedDirectory = self.getDirectory(directory, REMIXED)
        self.files = []
        self.size = -1
        self.batchSize = batchSize
        for file in listdir(self.originalDirectory):
            fullpath = join(self.originalDirectory, file)
            if isfile(fullpath) and supported(fullpath):
                self.files.append(fullpath)
        logging.info('%d files detected' % len(self.files))

    def getVectorSize(self):
        """Returns the maximum vector size from all songs in the corpus.

        :returns: Maximum vector size from all songs in this corpus.
        """
        if self.size != -1:
            return self.size
        size = 0
        for file in self.files:
            remixed = join(self.remixedDirectory, basename(file))
            size = max(size, len(self.load(file)))
            size = max(size, len(self.load(remixed)))
        self.size = size
        return size

    def __iter__(self):
        """Iterator interface implementation.

        :returns: Reference to this instance.
        """
        self.current = 0
        return self

    def next(self):
        """Creates and returns the next batch.

        Created batch is a list of (original, remixed) vectors.

        :returns: Next batch if any.
        """
        if (self.current * self.batchSize) >= len(self.files):
            raise StopIteration()
        batch = []
        for i in xrange(self.batchSize):
            index = (self.current * self.batchSize) + i
            if index < len(self.files):
                file = self.files[index]
                original = self.load(file)
                remixed = self.load(join(self.remixedDirectory, basename(file)))
                batch.append((original, remixed))
        self.current = self.current + 1
        return batch
