__author__ = 'sichen'
import time

from lightCaffe.net import *


def sgd_optimization_mnist(learning_rate=0.01, n_epochs=30, data_set="../data/mnist.pkl.gz", batch_size=20):

    train_iteration = 1000
    val_iteration = 16*20

    net = Net('multi_layer_network_net.prototxt')
    net.init()

    print '... training the model'

    validation_frequency = 100
    print_frequency = 20

    #test_score = 0.
    start_time = time.clock()
    iteration = 0
    while iteration < train_iteration:

        net.forward_and_backward()
        net.update(learning_rate)

        iteration += 1
        if (iteration + 1) % print_frequency == 0:
            print 'iteration %i, loss %f' % (iteration, net.total_loss)
        if (iteration + 1) % validation_frequency == 0:
            validation_loss = 0.
            validation_error = 0.
            for i in xrange(val_iteration):
                loss, error = net.forward_val()
                validation_loss += loss
                validation_error += error
            validation_loss /= val_iteration
            validation_error /= val_iteration
            print 'iteration %i, validation loss %f, error %f %%' % (iteration,
                                                                 validation_loss,
                                                                 validation_error * 100)

    end_time = time.clock()
    print 'The code ran for %.1fs' % (end_time - start_time)


if __name__ == '__main__':
    sgd_optimization_mnist()





