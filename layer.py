__author__ = 'sichen'

import numpy as np
import scipy as sp


class Layer:
    def __init__(self, name="default"):
        self.name = name


class InnerProductLayer(Layer):
    def __init__(self, n_in, n_out, name="Inner Product Layer"):
        self.name = name
        self.W = np.random.randn(n_in, n_out)
        self.b = np.random.randn(n_out)
        self.n_in = n_in
        self.n_out = n_out

    def __debug_information__(self):
        print self.name + " with " + "n_in: " + str(self.n_in) + " and n_out: " + str(self.n_out)
        print "W:"
        print self.W
        print "b:"
        print self.b

    def forward(self, layer_in):
        layer_in_shape = layer_in.shape
        assert(layer_in_shape[0] > 0)
        assert(layer_in_shape[1] == self.n_in)
        return np.dot(layer_in, self.W) + self.b

