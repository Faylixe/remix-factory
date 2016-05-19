#!/usr/bin/python

import logging

from os import listdir, makedirs
from os.path import isfile, join, exists, splitext, basename
from scipy.io import wavfile
from pydub import AudioSegment

import configuration

# Enumeration of all file format supported by this script.
SUPPORTED_EXTENSION = ['.wav', '.mp3']

#  Audio file format converters.
CONVERTERS = {}
CONVERTERS['.mp3'] = lambda file: AudioSegment.from_mp3(file)

class Corpus:
    """ Class that defines a corpus with batch facilities. """

    def __init__(self, directory):
        """ Default constructor. Ensures given directory exist and crawls original song. """
        if not exists(directory):
            raise IOError('Directory %s does not exist' % directory)
        self.originalDirectory = join(directory, configuration.CORPUS_ORIGINAL)
        self.remixedDirectory = join(directory, configuration.CORPUS_REMIXED)
        # TODO : Ensures directory exists.
        self.files = []
        for file in listdir(self.originalDirectory):
            fullpath = join(self.originalDirectory, file)
            extension = file[-4:]
            if isfile(fullpath) and extension in SUPPORTED_EXTENSION:
                self.files.append(fullpath)
        logging.info('%d files detected' % len(self.files))

    def getVectorSize(self):
        """ Returns the maximum vector size from all songs in the corpus. """
        size = 0
        for file in self.files:
            remixed = join(self.remixedDirectory, basename(file))
            size = max(size, len(load(file)))
            size = max(size, len(load(remixed)))
        return size

    def startBatch(self, batchSize):
        """ Initializes batching process using given size for created batch. """
        self.current = 0
        self.batchSize = batchSize
        if batchSize > len(self.files):
            return 1
        n = len(self.files) / batchSize
        if len(self.files) % batchSize != 0:
            return n + 1
        return n

    def nextBatch(self):
        """ Creates and returns the next batch. """
        batch = []
        for i in xrange(self.batchSize):
            index = self.current + i
            if index < len(self.files):
                original = self.files[index]
                remixed = join(self.remixedDirectory, basename(original))
                # TODO : Normalizes vector using right zero padding.
                batch.append((load(original), load(remixed)))
        self.current = self.current + 1
        return batch

# TODO : Ensure rate is equals for all songs.
def load(file):
    """ Loads the given wave file numerical values. """
    extension = file[-4:]
    if not exists(file) or not extension in SUPPORTED_EXTENSION:
        raise IOError('%s file format is not supported' % extension)
    if extension != '.wav':
        file = convert(file)
    _, data = wavfile.read(file)
    return data# .astype('float32') / 32767.0 # TODO : Ensure normalization process is valid.

def save(vector, file):
    """ Saves the given vector to the given file. """
    # TODO : implement
    return

def convert(file):
    """ Convert the given file to a valid wav format and returns the path of the converted file. """
    if not exists(configuration.CONVERTION_DIRECTORY):
        makedirs(configuration.CONVERTION_DIRECTORY)
    extension = file[-4:]
    if not exists(file) or not extension in SUPPORTED_EXTENSION:
        raise IOError('%s file format is not supported' % extension)
    filename = splitext(basename(file))[0]
    path = join(configuration.CONVERTION_DIRECTORY, filename + '.wav')
    if (not exists(path)):
        logging.info("Converting file %s" % file)
        CONVERTERS[extension](file).export(path, format='wav')
    return path
