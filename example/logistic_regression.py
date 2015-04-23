__author__ = 'sichen'
import cPickle
import gzip
import os
import sys
import time

import numpy as np
from lightCaffe.layer import *

def load_data(data_set):
    print '... loading data'
    f = gzip.open(data_set, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, data_set="../data/mnist.pkl.gz", batch_size=600):

    data_sets = load_data(data_set)
    train_set_x, train_set_y = data_sets[0]
    valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]

    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    print '... initializing network'

    ip_layer = InnerProductLayer(batch_size, 28*28, 10)
    soft_max_layer = SoftMaxLayer(batch_size, 10)
    loss_layer = LossLayer(batch_size)





