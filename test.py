__author__ = 'sichen'
import numpy as np
import scipy as sp
from layer import *


def test_layer():
    layer1 = Layer()
    layer2 = Layer("linear")
    print layer1.name
    print layer2.name


def test_fullyconnectedlayer():
    layer = InnerProductLayer(10, 8)
    rand_in = np.random.randn(5, 10)
    print layer.forward(rand_in)

if __name__ == "__main__":
    test_layer()
    test_fullyconnectedlayer()