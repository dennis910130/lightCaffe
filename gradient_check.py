__author__ = 'sichen'
import numpy as np
import scipy as sp
from layer import *


def check_soft_max_loss_layer():
    layer = SoftMaxLayer(3, 5)
    btm_data = np.random.randn(3, 5)
    layer.forward(btm_data)
    label = np.array([1, 2, 0])
    loss_layer = CrossEntropyLossLayer(3, 5, label)
    loss_layer.forward(layer.top_data)
    loss_layer.backward()
    layer.backward(loss_layer.btm_diff)
    print layer.btm_diff
    numerical_gradient = np.zeros((3,5))
    total_loss = loss_layer.total_loss
    eps = 1e-5
    print eps
    for i in range(0, 3):
        for j in range(0, 5):
            btm_data[i,j] += eps
            layer.forward(btm_data)
            loss_layer.forward(layer.top_data)
            numerical_gradient[i,j] = (loss_layer.total_loss - total_loss)/eps
            btm_data[i,j] -= eps
    print numerical_gradient


if __name__ == '__main__':
    check_soft_max_loss_layer()