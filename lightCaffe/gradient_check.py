__author__ = 'sichen'
from lightCaffe.layer import *
from lightCaffe.util import *

def check_soft_max_layer():
    layer = SoftMaxLayer(3, 5)
    btm_data = np.random.randn(3, 5)
    layer.forward(btm_data)
    top_diff = np.random.randn(3, 5)
    layer.backward(top_diff)
    gradient = layer.btm_diff
    numerical_gradient = np.zeros((3, 5))
    top_data = layer.top_data
    eps = 1e-5
    for i in range(0, 3):
        for j in range(0, 5):
            btm_data[i, j] += eps
            layer.forward(btm_data)
            delta_top_data = layer.top_data - top_data
            delta_loss = np.sum(delta_top_data * top_diff)
            numerical_gradient[i, j] = delta_loss / eps
            btm_data[i, j] -= eps
    print "SOFT MAX LOSS LAYER:"
    if is_near_enough(gradient, numerical_gradient):
        print "passed!"
    else:
        print "failed!"


def check_cross_entropy_loss_layer():
    soft_max_layer = SoftMaxLayer(3, 5)
    btm_data = np.random.randn(3, 5)
    soft_max_layer.forward(btm_data)
    loss_layer = CrossEntropyLossLayer(3, 5)
    label = np.array([1, 2, 0])
    loss_layer.forward(soft_max_layer.top_data, label)
    loss_layer.backward()
    gradient = loss_layer.btm_diff
    numerical_gradient = np.zeros((3, 5))
    total_loss = loss_layer.total_loss
    eps = 1e-5
    for i in range(0, 3):
        for j in range(0, 5):
            soft_max_layer.top_data[i,j] += eps
            loss_layer.forward(soft_max_layer.top_data, label)
            numerical_gradient[i,j] = (loss_layer.total_loss - total_loss)/eps
            soft_max_layer.top_data[i,j] -= eps
    print 'Cross Entropy Layer:'
    if is_near_enough(gradient, numerical_gradient):
        print 'passed!'
    else:
        print 'failed!'


def check_inner_product_layer():
    layer = InnerProductLayer(5, 10, 8)
    btm_data = np.random.randn(5, 10)
    layer.forward(btm_data)
    top_diff = np.random.randn(5, 8)
    layer.backward(top_diff)
    top_data = layer.top_data
    eps = 1e-5

    btm_diff = layer.btm_diff
    W_diff = layer.W_diff
    b_diff = layer.b_diff

    numerical_btm_diff = np.zeros((5, 10))
    numerical_W_diff = np.zeros((10, 8))
    numerical_b_diff = np.zeros((8, ))

    for i in range(0, 5):
        for j in range(0, 10):
            btm_data[i, j] += eps
            layer.forward(btm_data)
            delta_top_data = layer.top_data - top_data
            delta_loss = np.sum(delta_top_data * top_diff)
            numerical_btm_diff[i, j] = delta_loss / eps
            btm_data[i, j] -= eps
    for i in range(0, 10):
        for j in range(0, 8):
            layer.W[i, j] += eps
            layer.forward(btm_data)
            delta_top_data = layer.top_data - top_data
            delta_loss = np.sum(delta_top_data * top_diff)
            numerical_W_diff[i, j] = delta_loss / eps
            layer.W[i, j] -= eps
    for i in range(0, 8):
        layer.b[i] += eps
        layer.forward(btm_data)
        delta_top_data = layer.top_data - top_data
        delta_loss = np.sum(delta_top_data * top_diff)
        numerical_b_diff[i] = delta_loss / eps
        layer.b[i] -= eps

    print "INNER PRODUCT LAYER:"
    if is_near_enough(numerical_btm_diff, btm_diff) \
       and is_near_enough(numerical_W_diff, W_diff) \
       and is_near_enough(numerical_b_diff, b_diff):
        print "passed!"
    else:
        print "failed!"


def check_conv_layer():
    layer = ConvLayer(3, 4, 8, 1, 3, 5)
    btm_data = np.random.randn(3, 4, 8, 8)
    layer.forward(btm_data)
    top_diff = np.random.randn(3, 5, 8, 8)
    layer.backward(top_diff)
    top_data = layer.top_data
    eps = 1e-5

    btm_diff = layer.btm_diff
    W_diff = layer.W_diff
    b_diff = layer.b_diff

    numerical_btm_diff = np.zeros((3, 4, 8, 8))
    numerical_W_diff = np.zeros((5, 4, 3, 3))
    numerical_b_diff = np.zeros((5, ))

    for n in range(0, 3):
        for c in range(0, 4):
            for a in range(0, 8):
                for b in range(0, 8):
                    btm_data[n, c, a, b] += eps
                    layer.forward(btm_data)
                    delta_top_data = layer.top_data - top_data
                    delta_loss = np.sum(delta_top_data * top_diff)
                    numerical_btm_diff[n, c, a, b] = delta_loss / eps
                    btm_data[n, c, a, b] -= eps
    for n in range(0, 5):
        for c in range(0, 4):
            for a in range(0, 3):
                for b in range(0, 3):
                    layer.W[n, c, a, b] += eps
                    layer.forward(btm_data)
                    delta_top_data = layer.top_data - top_data
                    delta_loss = np.sum(delta_top_data * top_diff)
                    numerical_W_diff[n, c, a, b] = delta_loss / eps
                    layer.W[n, c, a, b] -= eps
    for i in range(0, 5):
        layer.b[i] += eps
        layer.forward(btm_data)
        delta_top_data = layer.top_data - top_data
        delta_loss = np.sum(delta_top_data * top_diff)
        numerical_b_diff[i] = delta_loss / eps
        layer.b[i] -= eps
    print "CONV LAYER:"
    if is_near_enough(numerical_btm_diff, btm_diff) \
       and is_near_enough(numerical_W_diff, W_diff) \
       and is_near_enough(numerical_b_diff, b_diff):
        print "passed!"
    else:
        print "failed!"


def check_relu_layer():
    layer = ReLULayer(3, 5)
    btm_data = np.random.randn(3, 5)
    layer.forward(btm_data)
    top_data = layer.top_data
    top_diff = np.random.randn(3, 5)
    layer.backward(top_diff)
    numerical_gradient = np.zeros((3,5))
    gradient = layer.btm_diff
    eps = 1e-5
    for i in range(0, 3):
        for j in range(0, 5):
            btm_data[i, j] += eps
            layer.forward(btm_data)
            delta_top_data = layer.top_data - top_data
            delta_loss = np.sum(delta_top_data * top_diff)
            numerical_gradient[i, j] = delta_loss / eps
            btm_data[i, j] -= eps
    print "ReLU LAYER:"
    if is_near_enough(numerical_gradient, gradient):
        print "passed!"
    else:
        print "failed!"

if __name__ == '__main__':
    check_cross_entropy_loss_layer()
    check_soft_max_layer()
    check_inner_product_layer()
    check_relu_layer()
    check_conv_layer()