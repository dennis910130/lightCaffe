__author__ = 'sichen'

import numpy as np
import scipy as sp


class Layer:
    def __init__(self, name="default"):
        self.name = name


class LossLayer(Layer):
    def __init__(self, n_batch, n_class):
        self.name = "generic loss layer"
        self.label = np.zeros((n_batch,), np.int16)
        self.n_class = n_class
        self.n_batch = n_batch
        self.btm_data = np.zeros((n_batch, n_class))
        self.btm_diff = np.zeros_like(self.btm_data)
        self.loss = np.zeros((n_batch,))
        self.total_loss = 0.0
        self.prediction = np.zeros((n_batch,), np.int16)


class InnerProductLayer(Layer):
    def __init__(self, n_batch, n_in, n_out, name="Inner Product Layer"):
        self.name = name
        self.W = np.random.randn(n_in, n_out) / 1e3
        self.b = np.random.randn(n_out) / 1e3
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.btm_data = np.zeros((n_batch, n_in))
        self.top_data = np.zeros((n_batch, n_out))
        self.btm_diff = np.zeros_like(self.btm_data)
        self.W_diff = np.zeros_like(self.W)
        self.b_diff = np.zeros_like(self.b)

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
        self.scale_data = np.zeros((n_batch,))

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

    def backward(self, top_diff):
        self.btm_diff = top_diff
        self.scale_data = np.sum(top_diff * self.top_data, axis=1)
        self.btm_diff = self.btm_diff - self.scale_data.reshape(self.n_batch, 1)
        self.btm_diff = self.btm_diff * self.top_data


class CrossEntropyLossLayer(LossLayer):
    def __init__(self, n_batch, n_class, name="Cross Entropy Loss Layer"):
        LossLayer.__init__(self, n_batch, n_class)
        self.name = name

    def forward(self, btm_data, label):
        self.label = label
        self.btm_data = btm_data
        self.loss = - np.log(btm_data)[np.arange(self.n_batch), self.label]
        self.total_loss = np.sum(self.loss)

    def backward(self):
        mask = np.zeros_like(self.btm_data)
        mask[np.arange(self.n_batch), self.label] = 1
        self.btm_diff = - 1 / self.btm_data * mask

    def error(self):
        self.prediction = np.argmax(self.btm_data, axis=1)
        return np.mean(np.not_equal(self.prediction, self.label))


class ReLULayer(Layer):
    def __init__(self, n_batch, n_in, name="ReLU Layer"):
        self.name = name
        self.n_batch = n_batch
        self.n_in = n_in
        self.btm_data = np.zeros((n_batch, n_in))
        self.top_data = np.zeros((n_batch, n_in))
        self.btm_diff = np.zeros_like(self.btm_data)
        self.top_diff = np.zeros_like(self.top_data)

    def forward(self, btm_data):
        self.btm_data = btm_data
        self.top_data = self.btm_data * (self.btm_data > 0)

    def backward(self, top_diff):
        self.top_diff = top_diff
        self.btm_diff = 1. * (self.top_diff > 0)