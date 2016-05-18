#!/usr/bin/python

import argparse
import logging
from neuronalnetwork import NeuronalNetwork

LOGGER = logging.getLogger('remix-factory')

def train(corpus, path):
    """ """
    LOGGER.info("Load corpus from directory %s" % args.dataset)
    corpus = corpus.build()
    n = 0
    model = NeuronalNetwork(n)
    LOGGER.info("Start training")
    model.train(corpus)
    LOGGER.info("Saving model to %s" % args.model)
    model.save(args.model)

def create(modelPath, songPath):
    """ """
    model = NeuronalNetwork.load(modelPath)
    # TODO : Load song as vector.
    remixed = model.apply(original)
    # TODO : Save generated song.

if __name__ == '__main__':
    # Parse command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', help='Trains a model using a given corpus', action='store_true')
    parser.add_argument('-c', '--create', help='Creates a remix from the given song using the given model', action='store_true')
    parser.add_argument('-m', '--model', help='Path of the model file to load or save')
    parser.add_argument('-d', '--dataset', help='Path of the corpus directory to use for training the model')
    args = parser.parse_args()
    # Handles action.
    if args.train:
        train(args.dataset, args.model)
    elif args.create:
        create(args.model) # TODO : Add song path.
    else:
        # TODO : Show error.
