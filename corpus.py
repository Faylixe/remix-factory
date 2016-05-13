#!/usr/bin/python

from os import listdir
from os.path import isfile, join, exists
from scipy.io import wavfile

import configuration

# Enumeration of all file format supported by this script.
SUPPORTED_EXTENSION = ['.wav']

#  Audio file format converters.
CONVERTERS = {}
CONVERTERS['.mp3'] = lambda file: return file

# Builds and returns the corpus from the directory retrieved from the configuration.
def get():
    return fromDirectory(configuration.CORPUS_DIRECTORY)

# Builds and returns the corpus from the given directory.
def fromDirectory(directory):
    if not exists(directory):
        raise IOError('Directory %s does not exist' % directory)
    corpus = []
    for file in listdir(directory):
        fullpath = join(directory, file)
        extension = file[-4:]
        if isfile(fullpath) and extension in SUPPORTED_EXTENSION:
            corpus.append(fromFile(fullpath))

#
def fromFile(file):
    extension = file[-4:]
    if not exists(file) or not extension in SUPPORTED_EXTENSION:
        raise IOError('%s file format is not supported' % extension)
    if extension != '.wav':
        file = convert(file)
    rate, data = wavfile.read(file)
    return data.astype('float32') / 32767.0, rate

# Convert the given file to a valid wav format and returns the path of the converted file.
def convert(file):
    extension = file[-4:]
    if not exists(file) or not extension in SUPPORTED_EXTENSION:
        raise IOError('%s file format is not supported' % extension)
    CONVERTERS[extension](file)
    return
