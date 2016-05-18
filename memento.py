#!/usr/bin/python

import pickle

class Memento:
    """ """

    def __init__(self, path, type='train'):
        """ Default constructor. """
        self.level = 0
        self.type = type
        self.path = path

    def isCorpusCreated(self):
        """ Indicates if the corpus has been created. """
        return self.level > 0

    def isNetworkCreated(self):
        """ Indicates if the neuronal network has been created. """
        return self.level > 1

    def notifyCorpusCreated(self):
        """ """
        self.level = 1
        self.save()

    def notifyNetworkCreated(self):
        """ """
        self.level = 2
        self.save()

    def save(self):
        """ Saves this memento to the given path. """
        stream = open(self.path, 'wb')
        pickle.dump(self, stream, -1)

    def load(path):
        """ Loads and returns the memento denoted by the given path. """
        stream = open(path, 'rb')
        return pickle.load(stream)
