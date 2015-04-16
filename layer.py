__author__ = 'sichen'

import numpy as np
import scipy as sp


class Layer:
    def __init__(self, name="default"):
        self.name = name


class InnerProductLayer(Layer):
    def __init__(self, n_batch, n_in, n_out, name="Inner Product Layer"):
        self.name = name
        self.W = np.random.randn(n_in, n_out)
        self.b = np.random.randn(n_out)
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.btm_data = np.zeros((n_batch, n_in))
        self.top_data = np.zeros((n_batch, n_out))
        self.btm_diff = np.zeros_like(self.btm_data)
        self.W_diff = np.zeros_like(self.W)
        self.b_diff = np.zeros_like(self.b_diff)

    def __debug_information__(self):
        print "name:"
        print self.name
        print "n_batch:"
        print self.n_batch
        print "n_in:"
        print self.n_in
        print "n_out:"
        print self.n_out
        print "W:"
        print self.W
        print "b:"
        print self.b

    def forward(self, btm_data):
        self.btm_data = btm_data
        btm_data_shape = self.btm_data.shape
        assert(btm_data_shape[0] > 0)
        assert(btm_data_shape[1] == self.n_in)
        self.top_data = np.dot(btm_data, self.W) + self.b

    def backward(self, top_diff):
        self.W_diff = np.dot(self.btm_data.T, top_diff)
        self.b_diff = np.dot(np.ones(self.btm_data.shape[0],), top_diff)
        self.btm_diff = np.dot(top_diff, self.W.T)

    def update(self, alpha_w, alpha_b):
        self.W = self.W - alpha_w * self.W_diff
        self.b = self.b - alpha_b * self.b_diff


class SoftMaxLayer(Layer):
    def __init__(self, n_batch, n_in, name="SoftMax Layer"):
        self.name = name
        self.n_batch = n_batch
        self.n_in = n_in
        self.btm_data = np.zeros((n_batch, n_in))
        self.top_data = np.zeros_like(self.btm_data)
        self.btm_diff = np.zeros_like(self.btm_data)

    def __debug_information__(self):
        print "name:"
        print self.name
        print "n_in:"
        print self.n_in

    def forward(self, btm_data):
        self.btm_data = btm_data
        maxes = np.amax(self.btm_data, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(self.btm_data - maxes)
        self.top_data = e / np.sum(e, axis=1).reshape(maxes.shape[0], 1)


