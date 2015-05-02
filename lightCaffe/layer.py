__author__ = 'sichen'

import numpy as np
import scipy as sp


class Layer:
    def __init__(self, n_in, n_out, n_batch, name="default"):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.n_batch = n_batch
        self.btm_data = None
        self.top_data = None
        self.btm_diff = None
        self.need_update = False

    def print_information(self):
        print '----------------------------------------------------------------------------------'
        print "%25s: n_batch %4i, n_in %4i, n_out %4i, need_update %r" \
              % (self.name, self.n_batch, self.n_in, self.n_out, self.need_update)
        print '----------------------------------------------------------------------------------'


class LossLayer(Layer):
    def __init__(self, n_in, n_out, n_batch, n_class, name="Loss Layer"):
        Layer.__init__(self, n_in, n_out, n_batch, name)
        self.label = None
        self.n_class = n_class
        self.loss = None
        self.total_loss = 0.0
        self.prediction = None


class InnerProductLayer(Layer):
    def __init__(self, n_batch, n_in, n_out, name="Inner Product Layer"):
        Layer.__init__(self, n_in, n_out, n_batch, name)
        self.W = np.random.randn(n_in, n_out) / 1e3
        self.b = np.random.randn(n_out) / 1e3
        self.W_diff = None
        self.b_diff = None
        self.need_update = True

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
        Layer.__init__(self, n_in, n_in, n_batch, name)
        self.scale_data = None

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
        LossLayer.__init__(self, n_class, n_class, n_batch, n_class, name)

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
        Layer.__init__(self, n_in, n_in, n_batch, name)

    def forward(self, btm_data):
        self.btm_data = btm_data
        self.top_data = self.btm_data * (self.btm_data > 0)

    def backward(self, top_diff):
        self.btm_diff = 1. * (self.btm_data > 0) * top_diff