__author__ = 'sichen'
import time

from lightCaffe.layer import *


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=30, data_set="../data/mnist.pkl.gz", batch_size=600):

    learning_rate /= batch_size

    print '... initializing network'

    data_layer = PklDataLayer(batch_size, data_set)
    data_layer.load_data()
    data_layer.print_information()
    ip_layer = InnerProductLayer(batch_size, 28*28, 10)
    ip_layer.print_information()
    soft_max_layer = SoftMaxLayer(batch_size, 10)
    soft_max_layer.print_information()
    loss_layer = CrossEntropyLossLayer(batch_size, 10)
    loss_layer.print_information()

    print '... training the model'

    validation_frequency = data_layer.n_train_batches
    print_frequency = 20

    #test_score = 0.
    start_time = time.clock()

    while data_layer.epoch_index < n_epochs:

        (data_input, label) = data_layer.get_next_batch_train()
        ip_layer.forward(data_input)
        soft_max_layer.forward(ip_layer.top_data)
        loss_layer.forward(soft_max_layer.top_data, label)

        loss_layer.backward()
        soft_max_layer.backward(loss_layer.btm_diff)
        ip_layer.backward(soft_max_layer.btm_diff)
        ip_layer.update(learning_rate, learning_rate)

        iteration = (data_layer.epoch_index - 1) * data_layer.n_train_batches + data_layer.batch_index_train
        if (iteration + 1) % print_frequency == 0:
            print 'epoch %i, mini_batch %i/%i, loss %f' % (data_layer.epoch_index,
                                                           data_layer.batch_index_train,
                                                           data_layer.n_train_batches,
                                                           loss_layer.total_loss)
        if (iteration + 1) % validation_frequency == 0:
            validation_loss = 0.
            validation_error = 0.
            for i in xrange(data_layer.n_val_batches):
                (data_input, label) = data_layer.get_next_batch_val()

                ip_layer.forward(data_input)
                soft_max_layer.forward(ip_layer.top_data)
                loss_layer.forward(soft_max_layer.top_data, label)
                validation_loss += loss_layer.total_loss
                validation_error += loss_layer.error()
            validation_loss /= data_layer.n_val_batches
            validation_error /= data_layer.n_val_batches
            print 'epoch %i, validation loss %f, error %f %%' % (data_layer.epoch_index,
                                                                 validation_loss,
                                                                 validation_error * 100)

    end_time = time.clock()
    print 'The code ran for %.1fs' % (end_time - start_time)


if __name__ == '__main__':
    sgd_optimization_mnist()




