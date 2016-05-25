#!/usr/bin/python

import argparse
import logging
from sys import exit

from dataset.batchiterator import BatchIterator
from model.neuronalnetwork import NeuronalNetwork

""" Default size for the batch size. """
DEFAULT_BATCH_SIZE = 10

""" Default learning rate value. """
DEFAULT_LEARNING_RATE = 1

def getDataset(datasetDirectory):
    """Creates and returns dataset from the given directory.

    :returns: Created dataset instance as BatchIterator object.
    """
    logging.info('Load dataset from directory %s' % datasetDirectory)
    return BatchIterator(datasetDirectory)

def create(datasetDirectory, modelDirectory, window):
    """Creates empty neuronal network using given parameters.

    :param datasetDirectory: Path of the directory to extract dataset from.
    :param modelDirectory: Path of the directory to save built model in.
    :param window: Window size to use for each neuron.
    :returns: Loaded dataset and created model.
    """
    dataset = getDataset(datasetDirectory)
    logging.info('Retrieving maximum vector size from dataset')
    size = dataset.getVectorSize()
    logging.info('Neuronal network size : %d' % size)
    model = NeuronalNetwork(modelDirectory, size, window)
    if window == None:
        window = size
    logging.info('Creating empty neuronal network with window size %d' % window)
    model.create()
    model.save()
    return dataset, model

def prepare(datasetDirectory, modelDirectory, window, shouldCreate):
    """Prepares training by loading dataset, and model.

    :param datasetDirectory: Path of the directory to extract dataset from.
    :param modelDirectory: Path of the directory to save built model in.
    :param window: Window size to use for each neuron.
    :param shouldCreate: Boolean flag that indicates if model should be created or loaded.
    :returns: Dataset and Model to use for training.
    """
    if shouldCreate:
        return create(datasetDirectory, modelDirectory, window)
    return getDataset(datasetDirectory), NeuronalNetwork.load(modelDirectory)

def train(datasetDirectory, modelDirectory, batchSize, learningRate, window, thread, shouldCreate):
    """ Performs a training step over the given model.

    Model could be created or simply loaded depending of the shouldCreate flag.

    :param datasetDirectory: Path of the directory to extract dataset from.
    :param modelDirectory: Path of the directory to save / load built model in.
    :param batchSize: (Optional) Size for batch (default to 10)
    :param learningRate: (Optional) Learning rate value (default to 1)
    :param window: Window size to use for each neuron.
    :param thread: Number of thread to use for creating model.
    :param shouldCreate: Boolean flag that indicates if model should be created or loaded.
    """
    dataset, model = prepare(datasetDirectory, modelDirectory, window, shouldCreate)
    if batchSize == None:
        batchSize = DEFAULT_BATCH_SIZE
    if learningRate == None:
        learningRate = DEFAULT_LEARNING_RATE
    i = 0
    for batch in dataset:
        logging.info('Training over batch #%d' % i)
        model.train(batch, learningRate)
        i += 1
    logging.info('Training step complete')

def generate(modelDirectory, songPath, remixPath):
    """Creates a song from the given model and save it.

    :param modelDirectory: Path of the directory to load model from.
    :param songPath: Path of the song file to remix.
    :param remixPath: Path of the output file to write.
    """
    logging.load('Loading model from %s' % modelDirectory)
    model = NeuronalNetwork.load(modelDirectory)
    logging.info('Loading %s as numerical vector' % songPath)
    vector = BatchIterator.load(songPath)
    logging.info('Applying neuronal network model to %s' % songPath)
    remixed = model.apply(vector) # TODO : Normalizes output vector ?
    logging.info('Saving created song to %s' % remixPath)
    wavfile.write(remixPath, 44100, remixed) # TODO : Get rate ?

def check(args, key):
    """Ensure parameter denoted from the given key exist in the args dictionary.

    :param args: Parsed command line arguments.
    :param key: Key to check.
    """
    if args.__dict__[key] == None:
        logging.error('Missing --%s parameter, abort' % key)
        exit(2)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--create', help='Creates a new neuronal network ready for training', action='store_true')
    parser.add_argument('-t', '--train', help='Trains a model using a given corpus', action='store_true')
    parser.add_argument('-r', '--remix', help='Generates a remix from the given song', action='store_true')
    parser.add_argument('-m', '--model', help='Path of the directory that will contains our model')
    parser.add_argument('-d', '--dataset', help='Path of the corpus directory to use for training the model')
    parser.add_argument('-s', '--song', help='Path of the song file to create remix for')
    parser.add_argument('-o', '--output', help='Path of the output remixed song file to create')
    parser.add_argument('-b', '--batchSize', type=int, help='(optional) Size of the batch to use for training, default value is 10')
    parser.add_argument('-l', '--learningRate', help='(optional) Learning rate parameter for gradient descent algorithm, default value is 1')
    parser.add_argument('-p', '--thread', type=int, help='(optional) Number of thread to use, if not specified maximum number will be used')
    parser.add_argument('-w', '--window', type=int, help='(optional) Size of the neuronal window to use, if not specified maximum number will be used')
    args = parser.parse_args()
    check(args, 'model')
    if args.train:
        check(args, 'dataset')
        train(args.dataset, args.model, args.batchSize, args.learningRate, args.window, args.thread, args.create)
    elif args.create and not args.train:
        check(args, 'dataset')
        create(args.dataset, args.model, args.window)
    elif args.remix:
        check(args, 'song')
        check(args, 'output')
        generate(args.model, args.song, args.output)
    else:
        logging.error('Missing action parameter')
        logging.error('--create, --train, or --remix action should be provided, abort')
        exit(2)
