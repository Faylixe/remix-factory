#!/usr/bin/python

import argparse
import logging

import configuration

from corpus import Corpus
from neuronalnetwork import NeuronalNetwork
from memento import Memento

def getCorpus(corpusDirectory):
    """ Retrieves a Corpus instance from the given directory. """
    logging.info('Load corpus from directory %s' % corpusDirectory)
    return Corpus(corpusDirectory)

def create(corpusDirectory, modelDirectory, thread):
    """ Creates an empty model using the given corpus. """
    corpus = getCorpus(corpusDirectory)
    logging.info('Retrieving maximum vector size from dataset')
    size = corpus.getVectorSize()
    logging.info('Neuronal network size : %d' % size)
    logging.info('Creating empty neuronal network')
    model = NeuronalNetwork(size, modelDirectory, thread)
    model.create()

def train(corpusDirectory, modelDirectory, batchSize, learningRate, thread):
    """ Train a new model using the given corpus directory and saves it to the given model path. """
    corpus = getCorpus(corpusDirectory)
    model = NeuronalNetwork(size, modelDirectory, thread)
    logging.info('Start training')
    # TODO : Add memento pattern for saving training step
    if batchSize == None:
        batchSize = configuration.DEFAULT_BATCH_SIZE
    if learningRate == None:
        batchSize = configuration.DEFAULT_LEARNING_RATE
    n = corpus.startBatch(batchSize)
    for i in xrange(n):
        logging.info('Training over batch #%d' % i)
        batch = corpus.nextBatch()
        model.train(batch, learningRate)
    logging.info('Training complete')

def generate(modelPath, songPath):
    """ Creates a song from the given model and save it. """
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--create', help='Creates a new neuronal network ready for training', action='store_true')
    parser.add_argument('-t', '--train', help='Trains a model using a given corpus', action='store_true')
    parser.add_argument('-r', '--remix', help='Generates a remix from the given song', action='store_true')
    parser.add_argument('-m', '--model', help='Path of the directory that will contains our model')
    parser.add_argument('-d', '--dataset', help='Path of the corpus directory to use for training the model')
    parser.add_argument('-b', '--batchSize', type=int, help='(optional) Size of the batch to use for training, default value is 10')
    parser.add_argument('-l', '--learningRate', help='(optional) Learning rate parameter for gradient descent algorithm, default value is 1')
    parser.add_argument('-p', '--thread', type=int, help='(optional) Number of thread to use, if not specified maximum number will be used.')
    args = parser.parse_args()
    if args.model == None:
        logging.error('Missing --model parameter, abort')
    if args.create:
        if args.dataset == None:
            logging.error('Missing --corpus parameter, abort')
        create(args.dataset, args.model, args.thread)
    elif args.train:
        if args.dataset == None:
            logging.error('Missing --corpus parameter, abort')
        train(args.dataset, args.model, args.batchSize, args.learningRate, args.thread)
    elif args.remix:
        logging.warning("Not implemented yet")
    else:
        # TODO : Update message
        logging.error('Action train or create should be provided')
