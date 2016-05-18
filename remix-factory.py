#!/usr/bin/python

import argparse
import logging

import configuration

from corpus import Corpus
from neuronalnetwork import NeuronalNetwork
from memento import Memento

def getModel(memento, corpus):
    if not memento.isNetworkCreated():
        logging.info('Retrieving maximum vector size from dataset')
        size = corpus.getVectorSize()
        logging.info('Neuronal network size : %d' % size)
        logging.info('Creating empty neuronal network')
        model = NeuronalNetwork(size)
        memento.notifyNetworkCreated()
        return model
    else:
        # TODO : Load model
        return

# TODO : Add memento pattern for saving training step.
def train(corpusPath, modelPath, memento):
    """ Train a new model using the given corpus directory and saves it to the given model path. """
    logging.info('Load corpus from directory %s' % args.dataset)
    corpus = Corpus(corpusPath)
    memento.notifyCorpusCreated()
    model = getModel(memento, corpus)
    logging.info('Start training')
    n = corpus.startBatch(configuration.BATCH_SIZE)
    for i in xrange(n):
        logging.info('Training over batch #%d' % i)
        batch = corpus.nextBatch()
        model.train(batch, configuration.ALPHA)
    logging.info('Saving model to %s' % args.model)
    model.save(modelPath)

def create(modelPath, songPath):
    """ Creates a song from the given model and save it. """
    logging.info('Loading model from file %s' % modelPath)
    model = NeuronalNetwork.load(modelPath)
    logging.info('Loading song from file %s' % songPath)
    original = corpus.load(songPath)
    logging.info('Applying remix transformation')
    remixed = model.apply(original)
    corpus.save(remixed, 'remixed-song.wav') # TODO : Build filename.

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S')
    # Parse command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Trains a model using a given corpus', action='store_true')
    parser.add_argument('-c', '--create', help='Creates a remix from the given song using the given model', action='store_true')
    parser.add_argument('-m', '--model', help='Path of the model file to load or save')
    parser.add_argument('-d', '--dataset', help='Path of the corpus directory to use for training the model')
    args = parser.parse_args()
    # Handles action.
    if args.train:
        train(args.dataset, args.model, Memento('memento.mem'))
    elif args.create:
        create(args.model) # TODO : Add song path ?.
    else:
        logging.error('Action train or create should be provided')
