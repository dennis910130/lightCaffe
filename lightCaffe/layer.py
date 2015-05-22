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
        if isinstance(self.n_in, tuple):
            print '----------------------------------------------------------------------------------'
            print "%25s: n_batch %4i, n_channel %4i, height %4i, width %4i, out_channel %4i, " \
                  "out_height %4i, out_width %4i, need_update %r" \
                  % (self.name, self.n_batch, self.n_in[0], self.n_in[1], self.n_in[2],
                     self.n_out[0], self.n_out[1], self.n_out[2], self.need_update)
            print '----------------------------------------------------------------------------------'
        else:
            print '----------------------------------------------------------------------------------'
            print "%25s: n_batch %4i, n_in %4i, n_out %4i, need_update %r" \
                  % (self.name, self.n_batch, self.n_in, self.n_out, self.need_update)
            print '----------------------------------------------------------------------------------'


class ImageLayer(Layer):
    def __init__(self,  n_batch, n_in, out_channel, padding, stride, filter_size, name='Image Layer'):
        #n_in is a tuple with three entries
        self.n_in = n_in
        self.n_batch = n_batch
        self.n_channel = n_in[0]
        self.height = n_in[1]
        self.width = n_in[2]
        self.out_channel = out_channel
        self.padding = padding
        self.stride = stride
        self.filter_size = filter_size
        self.n_out = (out_channel, (n_in[1]+padding*2-filter_size)/stride+1, (n_in[2]+padding*2-filter_size)/stride+1)
        self.btm_data = None
        self.top_data = None
        self.btm_diff = None
        self.need_update = False
        self.name = name



class PoolingLayer(ImageLayer):
    def __init__(self, n_batch, n_in, padding, filter_size, stride, pooling_type='Max', name='Pooling Layer'):
        ImageLayer.__init__(self, n_batch, n_in, n_in[0], padding, stride, filter_size, name)
        self.out_size = self.n_out[1]
        self.top_data = np.zeros((n_batch, self.n_channel, self.out_size, self.out_size))
        self.name = name
        self.pooling_type = pooling_type
        self.padded_btm_data = None

    def forward(self, btm_data):
        self.padded_btm_data = im_pad_batch(btm_data, self.padding)
        if self.pooling_type is 'Max':
            for n in xrange(self.n_batch):
                for c in xrange(self.n_channel):
                    for a in xrange(self.out_size):
                        for b in xrange(self.out_size):
                            self.top_data[n, c, a, b] = np.amax(self.padded_btm_data[n, c,
                                                                a*self.stride:a*self.stride+self.filter_size,
                                                                b*self.stride:b*self.stride+self.filter_size])
        elif self.pooling_type is 'Ave':
            for n in xrange(self.n_batch):
                for c in xrange(self.n_channel):
                    for a in xrange(self.out_size):
                        for b in xrange(self.out_size):
                            self.top_data[n, c, a, b] = np.mean(self.padded_btm_data[n, c,
                                                                a*self.stride:a*self.stride+self.filter_size,
                                                                b*self.stride:b*self.stride+self.filter_size])
        else:
            raise ValueError('Currently we only support Max and Ave pooling.')

    def backward(self, top_diff):
        padded_btm_diff = np.zeros_like(self.padded_btm_data)
        if self.pooling_type is 'Max':
            for n in xrange(self.n_batch):
                for c in xrange(self.n_channel):
                    for a in xrange(self.out_size):
                        for b in xrange(self.out_size):
                            padded_btm_diff[n, c,
                            a*self.stride:a*self.stride+self.filter_size,
                            b*self.stride:b*self.stride+self.filter_size] += \
                            top_diff[n, c, a, b] * (self.padded_btm_data[n, c,
                                                    a*self.stride:a*self.stride+self.filter_size,
                                                    b*self.stride:b*self.stride+self.filter_size] ==
                                                    self.top_data[n, c, a, b])
        if self.padding is 0:
            self.btm_diff = padded_btm_diff
        else:
            self.btm_diff = padded_btm_diff[:, :, self.padding:-self.padding, self.padding:-self.padding]


class ConvLayer(ImageLayer):
    def __init__(self, n_batch, n_in, padding=0, filter_size=0, out_channel=0, stride=1, sigma=1, name='Conv Layer', layer_param=None):
        if layer_param is None:
            ImageLayer.__init__(self, n_batch, n_in, out_channel, padding, stride, filter_size, name)
            self.W = np.random.randn(out_channel, self.n_channel, filter_size, filter_size) * sigma
            self.b = np.random.randn(out_channel) * sigma
            self.W_diff = None
            self.b_diff = None
            self.reshaped_W = None
            self.reshaped_batch_data = None
            self.out_size = self.n_out[1]
            self.top_data = None
            self.reshaped_out = None
            self.need_update = True
        else:
            ImageLayer.__init__(self, n_batch, n_in, layer_param.conv_param.out_channel, layer_param.conv_param.padding,
                                layer_param.conv_param.stride, layer_param.conv_param.filter_size, layer_param.name)
            self.W = np.random.randn(out_channel, self.n_channel, filter_size, filter_size) * \
                     layer_param.conv_param.sigma
            self.b = np.random.randn(out_channel) * layer_param.conv_param.sigma
            self.W_diff = None
            self.b_diff = None
            self.reshaped_W = None
            self.reshaped_batch_data = None
            self.out_size = self.n_out[1]
            self.top_data = None
            self.reshaped_out = None
            self.need_update = True

    def forward(self, btm_data):
        padded_btm_data = im_pad_batch(btm_data, self.padding)
        self.reshaped_batch_data = im2col_batch(padded_btm_data, self.filter_size, self.stride)
        self.reshaped_W = self.W.reshape((self.W.shape[0], -1))
        self.reshaped_out = np.dot(self.reshaped_batch_data, self.reshaped_W.T)
        self.reshaped_out += self.b.reshape((1, -1))
        self.top_data = np.rollaxis(self.reshaped_out.reshape((self.n_batch, self.out_size, self.out_size, -1),
                                                              order='F'), 3, 1)

    def backward(self, top_diff):
        reshaped_top_diff = np.rollaxis(top_diff, 1, 4).reshape(self.reshaped_out.shape, order='F')
        reshaped_btm_diff = np.dot(reshaped_top_diff, self.reshaped_W)
        padded_btm_diff = col2im_batch(reshaped_btm_diff, self.filter_size, self.stride, self.height+self.padding*2,
                                       self.n_batch, self.n_channel)
        if self.padding is 0:
            self.btm_diff = padded_btm_diff
        else:
            self.btm_diff = padded_btm_diff[:, :, self.padding:-self.padding, self.padding:-self.padding]
        reshaped_W_diff = np.dot(reshaped_top_diff.T, self.reshaped_batch_data)
        self.W_diff = reshaped_W_diff.reshape(self.W.shape)
        self.b_diff = np.dot(reshaped_top_diff.T, np.ones(reshaped_top_diff.shape[0],))

    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.W_diff
        self.b = self.b - learning_rate * self.b_diff


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
    def __init__(self, n_batch, n_in, n_out=0, sigma=1, name="Inner Product Layer", layer_param=None):
        if layer_param is None:
            Layer.__init__(self, n_in, n_out, n_batch, name)
            self.W = np.random.randn(n_in, n_out) * sigma
            self.b = np.random.randn(n_out) * sigma
            self.W_diff = None
            self.b_diff = None
            self.need_update = True
        else:
            Layer.__init__(self, n_in, layer_param.inner_product_param.num_output, n_batch, layer_param.name)
            self.W = np.random.randn(self.n_in, self.n_out) * layer_param.inner_product_param.sigma
            self.b = np.random.randn(self.n_out) * layer_param.inner_product_param.sigma
            self.W_diff = None
            self.b_diff = None
            self.need_update = True

    def forward(self, btm_data):
        self.btm_data = btm_data
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