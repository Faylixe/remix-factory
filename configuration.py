#!/usr/bin/python

CONVERTION_DIRECTORY = '/tmp/remix-factory/'
CORPUS_ORIGINAL = 'original'
CORPUS_REMIXED = 'remixed'
NEURONS_DIRECTORY = 'neurons'
NEURONS_FILE_PREFIX = 'neuron-'
BATCH_SIZE = 100
ALPHA = 100
THREAD = 8
# Audio parameters.
SAMPLING_FREQUENCY = 44100
HIDDEN_DIMENSION_SIZE = 1024  # Number of hidden dimensions. For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
