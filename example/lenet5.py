__author__ = 'sichen'
import time

from lightCaffe.layer import *


def sgd_optimization_mnist(learning_rate=0.1, n_epochs=30, data_set="../data/mnist.pkl.gz", batch_size=500):

    print '... initializing network'

    data_layer = PklDataLayer(batch_size, data_set)
    data_layer.load_data()
    data_layer.print_information()
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

    validation_frequency = data_layer.n_train_batches
    print_frequency = 1

    #test_score = 0.
    start_time = time.clock()

    while data_layer.epoch_index < n_epochs:

        (data_input, label) = data_layer.get_next_batch_train()
        data_input = data_input.reshape((batch_size, 1, 28, 28))
        layer0.forward(data_input)
        sigmoid1.forward(layer0.top_data)
        layer1.forward(sigmoid1.top_data)
        layer2.forward(layer1.top_data)
        sigmoid2.forward(layer2.top_data)
        layer3.forward(sigmoid2.top_data)
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

        print ip_layer2.W

        print ip_layer2.W_diff

        ip_layer.update(learning_rate)
        ip_layer2.update(learning_rate)
        layer0.update(learning_rate)
        layer2.update(learning_rate)

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

                data_input = data_input.reshape((batch_size, 1, 28, 28))
                layer0.forward(data_input)
                sigmoid1.forward(layer0.top_data)
                layer1.forward(sigmoid1.top_data)
                layer2.forward(layer1.top_data)
                sigmoid2.forward(layer2.top_data)
                layer3.forward(sigmoid2.top_data)
                ip_layer.forward(layer3.top_data.reshape((batch_size, -1)))
                sigmoid3.forward(ip_layer.top_data)
                ip_layer2.forward(sigmoid3.top_data)
                soft_max_layer.forward(ip_layer2.top_data)
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
