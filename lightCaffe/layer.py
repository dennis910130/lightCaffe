__author__ = 'sichen'

import numpy as np
import scipy as sp
from math_util import *
import gzip
import cPickle


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


class ImageLayer:
    def __init__(self, n_batch, n_channel, height, width, name='Image Layer'):
        self.n_batch = n_batch
        self.n_channel = n_channel
        self.height = height
        self.width = width
        self.name = name
        self.btm_data = None
        self.top_data = None
        self.btm_diff = None
        self.need_update = False

    def print_information(self):
        print '----------------------------------------------------------------------------------'
        print "%25s: n_batch %4i, n_channel %4i, height %4i, width %4i" \
              % (self.name, self.n_batch, self.n_channel, self.height, self.width)
        print '----------------------------------------------------------------------------------'


class ConvLayer(ImageLayer):
    def __init__(self, n_batch, n_channel, height, width, padding, filter_size, out_channel, stride=1, name='Conv Layer'):
        ImageLayer.__init__(self, n_batch, n_channel, height, width)
        self.padding = padding
        self.stride = stride
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.W = np.random.randn(out_channel, n_channel, filter_size, filter_size)
        self.b = np.random.randn(out_channel)
        self.W_diff = None
        self.b_diff = None
        self.reshaped_W = None
        self.reshaped_batch_data = None
        self.out_size = (height + padding * 2 - filter_size) / stride + 1
        self.top_data = None
        self.reshaped_out = None

    def forward(self, btm_data):
        padded_btm_data = im_pad_batch(btm_data, self.padding)
        self.reshaped_batch_data = im2col_batch(padded_btm_data, self.filter_size, self.stride)
        self.reshaped_W = self.W.reshape((self.W.shape[0], -1))
        self.reshaped_out = np.dot(self.reshaped_batch_data, self.reshaped_W.T)
        self.reshaped_out += self.b.reshape((1, -1))
        self.top_data = np.rollaxis(self.reshaped_out.reshape((self.n_batch, self.out_size, self.out_size, -1),
                                                              order='F'), 3, 1)


class DataLayer:
    def __init__(self, n_batch, name='Data Layer'):
        self.name = name
        self.n_batch = n_batch
        self.need_update = False

    def print_information(self):
        print '----------------------------------------------------------------------------------'
        print "%25s: n_batch %4i" \
              % (self.name, self.n_batch)
        print '----------------------------------------------------------------------------------'


class PklDataLayer(DataLayer):
    def __init__(self, n_batch=0, pkl_path="", name='Pkl Data Layer', layer_param=None):
        if layer_param is None:
            DataLayer.__init__(self, n_batch, name)
            self.pkl_path = pkl_path
            self.train_set_x = None
            self.train_set_y = None
            self.test_set_x = None
            self.test_set_y = None
            self.val_set_x = None
            self.val_set_y = None
            self.n_train_batches = 0
            self.n_test_batches = 0
            self.n_val_batches = 0
            self.batch_index_train = 0
            self.batch_index_val = 0
            self.batch_index_test = 0
            self.n_out = 0
            self.epoch_index = 1
        else:
            DataLayer.__init__(self, layer_param.pkl_data_param.batch_size, layer_param.name)
            self.pkl_path = layer_param.pkl_data_param.source
            self.train_set_x = None
            self.train_set_y = None
            self.test_set_x = None
            self.test_set_y = None
            self.val_set_x = None
            self.val_set_y = None
            self.n_train_batches = 0
            self.n_test_batches = 0
            self.n_val_batches = 0
            self.batch_index_train = 0
            self.batch_index_val = 0
            self.batch_index_test = 0
            self.n_out = 0
            self.epoch_index = 1

    def load_data(self):
        print '... loading data'
        f = gzip.open(self.pkl_path, 'rb')
        train_set, val_set, test_set = cPickle.load(f)
        f.close()
        self.train_set_x, self.train_set_y = train_set
        self.test_set_x, self.test_set_y = test_set
        self.val_set_x, self.val_set_y = val_set
        self.n_train_batches = self.train_set_x.shape[0] / self.n_batch
        self.n_test_batches = self.test_set_x.shape[0] / self.n_batch
        self.n_val_batches = self.val_set_x.shape[0] / self.n_batch
        self.n_out = self.train_set_x.shape[1]

    def get_next_batch_train(self):
        batch_data = (self.train_set_x[self.batch_index_train * self.n_batch : (self.batch_index_train + 1) * self.n_batch], \
                      self.train_set_y[self.batch_index_train * self.n_batch : (self.batch_index_train + 1) * self.n_batch])
        self.batch_index_train += 1
        if self.batch_index_train >= self.n_train_batches:
            self.epoch_index += 1
            self.batch_index_train -= self.n_train_batches
        return batch_data

    def get_next_batch_test(self):
        batch_data = (self.test_set_x[self.batch_index_test * self.n_batch : (self.batch_index_test + 1) * self.n_batch], \
                      self.test_set_y[self.batch_index_test * self.n_batch : (self.batch_index_test + 1) * self.n_batch])
        self.batch_index_test += 1
        if self.batch_index_test >= self.n_test_batches:
            self.batch_index_test -= self.n_test_batches
        return batch_data

    def get_next_batch_val(self):
        batch_data = (self.val_set_x[self.batch_index_val * self.n_batch : (self.batch_index_val + 1) * self.n_batch], \
                      self.val_set_y[self.batch_index_val * self.n_batch : (self.batch_index_val + 1) * self.n_batch])
        self.batch_index_val += 1
        if self.batch_index_val >= self.n_val_batches:
            self.batch_index_val -= self.n_val_batches
        return batch_data


class LossLayer(Layer):
    def __init__(self, n_in, n_out, n_batch, n_class, name="Loss Layer"):
        Layer.__init__(self, n_in, n_out, n_batch, name)
        self.label = None
        self.n_class = n_class
        self.loss = None
        self.total_loss = 0.0
        self.prediction = None


class InnerProductLayer(Layer):
    def __init__(self, n_batch, n_in, n_out=0, name="Inner Product Layer", layer_param=None):
        if layer_param is None:
            Layer.__init__(self, n_in, n_out, n_batch, name)
            self.W = np.random.randn(n_in, n_out) / 1e3
            self.b = np.random.randn(n_out) / 1e3
            self.W_diff = None
            self.b_diff = None
            self.need_update = True
        else:
            Layer.__init__(self, n_in, layer_param.inner_product_param.num_output, n_batch, layer_param.name)
            self.W = np.random.randn(self.n_in, self.n_out) / 1e3
            self.b = np.random.randn(self.n_out) / 1e3
            self.W_diff = None
            self.b_diff = None
            self.need_update = True

    def forward(self, btm_data):
        self.btm_data = btm_data
        btm_data_shape = self.btm_data.shape
        self.top_data = np.dot(btm_data, self.W) + self.b

    def backward(self, top_diff):
        self.W_diff = np.dot(self.btm_data.T, top_diff)
        self.b_diff = np.dot(np.ones(self.btm_data.shape[0],), top_diff)
        self.btm_diff = np.dot(top_diff, self.W.T)

    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.W_diff
        self.b = self.b - learning_rate * self.b_diff


class SoftMaxLayer(Layer):
    def __init__(self, n_batch, n_in, name="SoftMax Layer", layer_param=None):
        if layer_param is None:
            Layer.__init__(self, n_in, n_in, n_batch, name)
            self.scale_data = None
        else:
            Layer.__init__(self, n_in, n_in, n_batch, layer_param.name)
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
    def __init__(self, n_batch, n_class, name="Cross Entropy Loss Layer", layer_param=None):
        if layer_param is None:
            LossLayer.__init__(self, n_class, n_class, n_batch, n_class, name)
        else:
            LossLayer.__init__(self, n_class, n_class, n_batch, n_class, layer_param.name)

    def forward(self, btm_data, label):
        self.label = label
        self.btm_data = btm_data
        self.loss = - np.log(btm_data)[np.arange(self.n_batch), self.label]
        self.total_loss = np.sum(self.loss) / self.n_batch

    def backward(self):
        mask = np.zeros_like(self.btm_data)
        mask[np.arange(self.n_batch), self.label] = 1.0 / self.n_batch
        self.btm_diff = - 1 / self.btm_data * mask

    def error(self):
        self.prediction = np.argmax(self.btm_data, axis=1)
        return np.mean(np.not_equal(self.prediction, self.label))


class ReLULayer(Layer):
    def __init__(self, n_batch, n_in, name="ReLU Layer", layer_param=None):
        if layer_param is None:
            Layer.__init__(self, n_in, n_in, n_batch, name)
        else:
            Layer.__init__(self, n_in, n_in, n_batch, layer_param.name)

    def forward(self, btm_data):
        self.btm_data = btm_data
        self.top_data = self.btm_data * (self.btm_data > 0)

    def backward(self, top_diff):
        self.btm_diff = 1. * (self.btm_data > 0) * top_diff