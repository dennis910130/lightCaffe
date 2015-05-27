__author__ = 'sichen'
import cPickle
import gzip
import time

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


def sgd_optimization_mnist(learning_rate=0.01, n_epochs=1000, data_set="../data/mnist.pkl.gz", batch_size=20, n_hidden=500):

    data_sets = load_data(data_set)
    train_set_x, train_set_y = data_sets[0]
    valid_set_x, valid_set_y = data_sets[1]
    test_set_x, test_set_y = data_sets[2]

    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    print '... initializing network'

    ip_layer_1 = InnerProductLayer(batch_size, 28*28, 500, sigma=5e-2)
    ip_layer_1.print_information()
    sigmoid_layer = SigmoidLayer(batch_size, 500)
    sigmoid_layer.print_information()
    ip_layer_2 = InnerProductLayer(batch_size, 500, 10, sigma=1e-1)
    ip_layer_2.print_information()
    soft_max_layer = SoftMaxLayer(batch_size, 10)
    soft_max_layer.print_information()
    loss_layer = CrossEntropyLossLayer(batch_size, 10)
    loss_layer.print_information()

    print '... training the model'

    validation_frequency = n_train_batches
    print_frequency = 20

    #test_score = 0.
    start_time = time.clock()
    epoch = 0

    while epoch < n_epochs:
        epoch += 1
        for mini_batch_index in xrange(n_train_batches):

            data_input = train_set_x[mini_batch_index * batch_size: (mini_batch_index + 1) * batch_size]
            label = train_set_y[mini_batch_index * batch_size: (mini_batch_index + 1) * batch_size]
            ip_layer_1.forward(data_input)
            sigmoid_layer.forward(ip_layer_1.top_data)
            ip_layer_2.forward(sigmoid_layer.top_data)
            soft_max_layer.forward(ip_layer_2.top_data)
            loss_layer.forward(soft_max_layer.top_data, label)

            loss_layer.backward()
            soft_max_layer.backward(loss_layer.btm_diff)
            ip_layer_2.backward(soft_max_layer.btm_diff)
            sigmoid_layer.backward(ip_layer_2.btm_diff)
            ip_layer_1.backward(sigmoid_layer.btm_diff)
            ip_layer_1.update(learning_rate)
            ip_layer_2.update(learning_rate)
            iteration = (epoch - 1) * n_train_batches + mini_batch_index
            if (iteration + 1) % print_frequency == 0:
                print 'epoch %i, mini_batch %i/%i, loss %f' % (epoch,
                                                                 mini_batch_index,
                                                                 n_train_batches,
                                                                 loss_layer.total_loss)
            if (iteration + 1) % validation_frequency == 0:
                validation_loss = 0.
                validation_error = 0.
                for i in xrange(n_valid_batches):
                    data_input = valid_set_x[i * batch_size: (i + 1) * batch_size]
                    label = valid_set_y[i * batch_size: (i + 1) * batch_size]

                    ip_layer_1.forward(data_input)
                    sigmoid_layer.forward(ip_layer_1.top_data)
                    ip_layer_2.forward(sigmoid_layer.top_data)
                    soft_max_layer.forward(ip_layer_2.top_data)
                    loss_layer.forward(soft_max_layer.top_data, label)
                    validation_loss += loss_layer.total_loss
                    validation_error += loss_layer.error()
                validation_loss /= n_valid_batches
                validation_error /= n_valid_batches
                print 'epoch %i, mini_batch %i/%i, validation loss %f, error %f %%' % (epoch,
                                                                                          mini_batch_index,
                                                                                          n_train_batches,
                                                                                          validation_loss,
                                                                                          validation_error * 100)

    end_time = time.clock()
    print 'The code ran for %.1fs' % (end_time - start_time)


if __name__ == '__main__':
    sgd_optimization_mnist()




