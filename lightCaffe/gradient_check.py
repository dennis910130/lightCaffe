__author__ = 'sichen'
from lightCaffe.layer import *


def check_soft_max_loss_layer():
    layer = SoftMaxLayer(3, 5)
    btm_data = np.random.randn(3, 5)
    layer.forward(btm_data)
    label = np.array([1, 2, 0])
    loss_layer = CrossEntropyLossLayer(3, 5, label)
    loss_layer.forward(layer.top_data, label)
    loss_layer.backward()
    layer.backward(loss_layer.btm_diff)
    numerical_gradient = np.zeros((3,5))
    total_loss = loss_layer.total_loss
    eps = 1e-5
    for i in range(0, 3):
        for j in range(0, 5):
            btm_data[i,j] += eps
            layer.forward(btm_data)
            loss_layer.forward(layer.top_data, label)
            numerical_gradient[i,j] = (loss_layer.total_loss - total_loss)/eps
            btm_data[i,j] -= eps
    print "SOFT MAX LOSS LAYER:"
    if np.linalg.norm(numerical_gradient - layer.btm_diff) < 1e-3:
        print "passed!"
    else:
        print "failed!"


def check_inner_product_layer():
    layer = InnerProductLayer(5, 10, 8)
    btm_data = np.random.randn(5, 10)
    layer.forward(btm_data)
    soft_max_layer = SoftMaxLayer(5, 8)
    soft_max_layer.forward(layer.top_data)
    label = np.array([1, 2, 0, 7, 6])
    loss_layer = CrossEntropyLossLayer(5, 8, label)
    loss_layer.forward(soft_max_layer.top_data, label)
    total_loss = loss_layer.total_loss
    eps = 1e-5
    loss_layer.backward()
    soft_max_layer.backward(loss_layer.btm_diff)
    layer.backward(soft_max_layer.btm_diff)

    W_diff = layer.W_diff
    b_diff = layer.b_diff

    numerical_btm_diff = np.zeros((5, 10))
    numerical_W_diff = np.zeros((10, 8))
    numerical_b_diff = np.zeros((8, ))

    for i in range(0, 5):
        for j in range(0, 10):
            btm_data[i,j] += eps
            layer.forward(btm_data)
            soft_max_layer.forward(layer.top_data)
            loss_layer.forward(soft_max_layer.top_data, label)
            numerical_btm_diff[i,j] = (loss_layer.total_loss - total_loss)/eps
            btm_data[i,j] -= eps
    for i in range(0, 10):
        for j in range(0, 8):
            layer.W[i,j] += eps
            layer.forward(btm_data)
            soft_max_layer.forward(layer.top_data)
            loss_layer.forward(soft_max_layer.top_data, label)
            numerical_W_diff[i,j] = (loss_layer.total_loss - total_loss)/eps
            layer.W[i,j] -= eps
    for i in range(0, 8):
        layer.b[i] += eps
        layer.forward(btm_data)
        soft_max_layer.forward(layer.top_data)
        loss_layer.forward(soft_max_layer.top_data, label)
        numerical_b_diff[i] = (loss_layer.total_loss - total_loss)/eps
        layer.b[i] -= eps

    print "INNER PRODUCT LAYER:"
    if np.linalg.norm(numerical_btm_diff - layer.btm_diff) < 1e-3 \
            and np.linalg.norm(numerical_W_diff - W_diff) < 1e-3 \
            and np.linalg.norm(numerical_b_diff - b_diff) < 1e-3:
        print "passed!"
    else:
        print "failed!"


if __name__ == '__main__':
    check_soft_max_loss_layer()
    check_inner_product_layer()