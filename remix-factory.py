#!/usr/bin/python

import argparse
import logging
from sys import exit

import configuration
from corpus import Corpus, load
from neuronalnetwork import NeuronalNetwork

def getCorpus(corpusDirectory):
    """ Retrieves a Corpus instance from the given directory. """
    logging.info('Load corpus from directory %s' % corpusDirectory)
    return Corpus(corpusDirectory)

def getVectorSize(corpus):
    """ Retrieves maximum vector size from the given corpus. """
    logging.info('Retrieving maximum vector size from dataset')
    return corpus.getVectorSize()

def create(corpusDirectory, modelDirectory, window, thread):
    """ Creates an empty model using the given corpus. """
    corpus = getCorpus(corpusDirectory)
    size = getVectorSize(corpus)
    logging.info('Neuronal network size : %d' % size)
    if window == None:
        window = size
    logging.info('Creating empty neuronal network with window size %d' % window)
    model = NeuronalNetwork(size, modelDirectory, window, thread)
    model.create()

def train(corpusDirectory, modelDirectory, batchSize, learningRate, window, thread):
    """ Train a new model using the given corpus directory and saves it to the given model path. """
    corpus = getCorpus(corpusDirectory)
    size = getVectorSize(corpus)
    model = NeuronalNetwork(size, modelDirectory, window, thread)
    logging.info('Start training')
    if batchSize == None:
        batchSize = configuration.DEFAULT_BATCH_SIZE
    if learningRate == None:
        batchSize = configuration.DEFAULT_LEARNING_RATE
    if window == None:
        window = size
    n = corpus.startBatch(batchSize)
    for i in xrange(n):
        logging.info('Training over batch #%d' % i)
        batch = corpus.nextBatch()
        model.train(batch, learningRate)
    logging.info('Training complete')

def generate(modelDirectory, songPath, remixPath):
    """ Creates a song from the given model and save it. """
    size = NeuronalNetwork.getSize(modelDirectory)
    model = NeuronalNetwork(size, modelDirectory, thread)
    logging.info('Loading %s as numerical vector' % songPath)
    vector = load(songPath)
    logging.info('Applying neuronal network model' % songPath)
    remixed = model.apply(vector)
    logging.info('Saving created song to %s' % remixPath)
    wavfile.write(remixPath, 44100, remixed) # TODO : Get rate ?

def check(args, key):
    """ Ensure parameter denoted from the given key exist in the args dictionary"""
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
    if args.create:
        check(args, 'dataset')
        create(args.dataset, args.model, args.window, args.thread)
    elif args.train:
        check(args, 'dataset')
        train(args.dataset, args.model, args.batchSize, args.learningRate, args.window, args.thread)
    elif args.remix:
        check(args, 'song')
        check(args, 'output')
        generate(args.model, args.song, args.output)
    else:
        logging.error('Missing action parameter')
        logging.error('--create, --train, or --remix action should be provided, abort')
        exit(2)
