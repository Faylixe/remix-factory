#!/usr/bin/python

# File parameters.
CORPUS_DIRECTORY = 'corpus/'
MODEL_NAME = 'apolon-model'
MODEL_FILE_PREFIX = 'apolon-model-'
MODEL_DIRECTORY = 'model/'
CONVERTION_DIRECTORY = '/tmp/apolon/'

# Audio parameters.
SAMPLING_FREQUENCY = 44100
HIDDEN_DIMENSION_SIZE = 1024  # Number of hidden dimensions. For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
