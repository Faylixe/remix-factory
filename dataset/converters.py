#!/usr/bin/python

import logging

from os import makedirs
from os.path import join, exists, splitext, basename
from pydub import AudioSegment

""" Target directory in which converted file will be written. """
CONVERTION_DIRECTORY = '/tmp/remix-factory/'

""" Enumeration of all file format supported by this script. """
SUPPORTED_EXTENSION = ['.wav', '.mp3']

""" Audio file format converters. """
CONVERTERS = {
    '.mp3': lambda file: AudioSegment.from_mp3(file)
}

def supported(file):
    """
    """
    return file[-4:] in SUPPORTED_EXTENSION

def convert(file):
    """Convert the given file to a valid wav format.

    :param file: File to convert.
    :returns: Path of the converted file.
    """
    extension = file[-4:]
    if extension == '.wav':
        return file
    if not exists(file):
        raise IOError('%s file not found' % file)
    if not extension in SUPPORTED_EXTENSION:
        raise IOError('%s file format is not supported' % file)
    if not exists(CONVERTION_DIRECTORY):
        makedirs(CONVERTION_DIRECTORY)
    filename = splitext(basename(file))[0]
    path = join(CONVERTION_DIRECTORY, filename + '.wav')
    if (not exists(path)):
        logging.info("Converting file %s" % file)
        CONVERTERS[extension](file).export(path, format='wav')
    return path
