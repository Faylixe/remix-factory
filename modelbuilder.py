#!/usr/bin/python

import tensorflow

#
class ModelBuilder:

    #
    def __init__(self):
        return

    #
    def build(self):
        x, W, b, y_ = self.createVariables()
        y = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)
        crossEntropy = tensorflow.reduce_mean(-tensorflow.reduce_sum(y_ * tensorflow.log(y), reduction_indices=[1]))
        trainStep = tensorflow.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)
        session = tensorflow.Session()
        session.run(tensorflow.initialize_all_variables())
        for i in range(1000): # TODO : Batch size
            xs, ys = None, None
            session.run(trainStep, feed_dict={x: xs, y_: ys})

    #
    def createVariables(self):
        x = tensorflow.placeholder(tensorflow.float32, [None, None]) # TODO : None, Song vector size.
        W = tensorflow.Variable(tensorflow.zeroes([None, None])) # TODO : Song vector size, Song vector size.
        b = tensorflow.Variable(tensorflow.zeroes([None])) # TODO : Song vector size
        y_ = tensorflow.placeholder(tensorflow.float32, [None, None]) # TODO : None, Song vector size
        return (x, W, b, y_)
