__author__ = 'sichen'
import time
import cv2
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


def sgd_optimization_mnist(learning_rate=0.01, n_epochs=30, data_set="../data/mnist.pkl.gz", batch_size=3):
    data_sets = load_data(data_set)
    train_set_x, train_set_y = data_sets[0]
    valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]

    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    print '... initializing network'

    layer0 = ConvLayer(batch_size, (1, 28, 28), 0, 5, 20, 1, 5e-2)
    layer0.print_information()
    sigmoid1 = SigmoidLayer(batch_size, (20, 24, 24))
    sigmoid1.print_information()
    layer1 = PoolingLayer(batch_size, (20, 24, 24), 0, 2, 2)
    layer1.print_information()
    layer2 = ConvLayer(batch_size, (20, 12, 12), 0, 5, 50, 1, 1e-1)
    layer2.print_information()
    sigmoid2 = SigmoidLayer(batch_size, (50, 8, 8))
    sigmoid2.print_information()
    layer3 = PoolingLayer(batch_size, (50, 8, 8), 0, 2, 2)
    layer3.print_information()

    ip_layer = InnerProductLayer(batch_size, 50*4*4, 500, 5e-2)
    ip_layer.print_information()
    sigmoid3 = SigmoidLayer(batch_size, 500)
    sigmoid3.print_information()
    ip_layer2 = InnerProductLayer(batch_size, 500, 10, 1e-1)
    ip_layer2.print_information()
    soft_max_layer = SoftMaxLayer(batch_size, 10)
    soft_max_layer.print_information()
    loss_layer = CrossEntropyLossLayer(batch_size, 10)
    loss_layer.print_information()

    print '... training the model'

    validation_frequency = 1
    print_frequency = 1
    n_valid_batches = 1

    #test_score = 0.
    start_time = time.clock()
    epoch = 0
    mini_batch_index = 0

    data_input = valid_set_x[mini_batch_index * batch_size: (mini_batch_index + 1) * batch_size]
    label = valid_set_y[mini_batch_index * batch_size: (mini_batch_index + 1) * batch_size]
    print data_input.reshape((batch_size, 1, 28, 28))[0,0,:]
    cv2.imwrite('a.png',data_input.reshape((batch_size, 1, 28, 28))[0,0,:]*255)
    print label
    data_input = data_input.reshape((batch_size, 1, 28, 28))
    layer0.forward(data_input)

    sigmoid1.forward(layer0.top_data)
    cv2.imwrite('b.png',sigmoid1.top_data[0,0,:]*255)
    layer1.forward(sigmoid1.top_data)
    cv2.imwrite('c.png', layer1.top_data[0,0,:]*255)
    layer2.forward(layer1.top_data)
    cv2.imwrite('d.png', layer2.top_data[0,0,:]*255)
    sigmoid2.forward(layer2.top_data)
    layer3.forward(sigmoid2.top_data)
    print layer3.top_data[1,:]
    print layer3.top_data.reshape((batch_size, -1))[1,:].shape
    print np.sum(layer3.top_data[1,:])
    print np.sum(layer3.top_data.reshape((batch_size, -1), order='F')[1,:])
    ip_layer.forward(layer3.top_data.reshape((batch_size, -1), order='F'))
    sigmoid3.forward(ip_layer.top_data)
    ip_layer2.forward(sigmoid3.top_data)
    soft_max_layer.forward(ip_layer2.top_data)
    loss_layer.forward(soft_max_layer.top_data, label)

    loss_layer.backward()
    soft_max_layer.backward(loss_layer.btm_diff)
    ip_layer2.backward(soft_max_layer.btm_diff)
    sigmoid3.backward(ip_layer2.btm_diff)
    ip_layer.backward(sigmoid3.btm_diff)
    layer3.backward(ip_layer.btm_diff.reshape((batch_size, 50, 4, 4), order='F'))
    sigmoid2.backward(layer3.btm_diff)
    layer2.backward(sigmoid2.btm_diff)
    layer1.backward(layer2.btm_diff)
    sigmoid1.backward(layer1.btm_diff)
    layer0.backward(sigmoid1.btm_diff)

    ip_layer.update(learning_rate)
    ip_layer2.update(learning_rate)
    layer0.update(learning_rate)
    layer2.update(learning_rate)


if __name__ == '__main__':
    sgd_optimization_mnist()
