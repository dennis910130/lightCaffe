__author__ = 'sichen'
from lightCaffe.layer import *
from lightCaffe.net import *

def test_layer():
    layer1 = Layer()
    layer2 = Layer("linear")
    print layer1.name
    print layer2.name


def test_inner_product_layer():
    layer = InnerProductLayer(5, 10, 8)
    btm_data = np.random.randn(5, 10)
    top_diff = np.random.randn(5, 8)
    layer.forward(btm_data)
    layer.backward(top_diff)
    print layer.top_data
    print layer.W_diff


def test_soft_max_layer():
    layer = SoftMaxLayer(3, 5)
    btm_data = np.random.randn(3, 5)
    layer.forward(btm_data)
    top_diff = np.random.randn(3, 5)
    layer.backward(top_diff)
    print layer.top_data
    print layer.btm_diff


def test_cross_entropy_layer():
    layer = SoftMaxLayer(3, 5)
    btm_data = np.random.randn(3, 5)
    layer.forward(btm_data)
    label = np.array([1, 3, 0])
    loss_layer = CrossEntropyLossLayer(3, 5)
    loss_layer.forward(layer.top_data, label)
    print loss_layer.loss
    print loss_layer.total_loss
    loss_layer.backward()
    print loss_layer.btm_diff
    print loss_layer.error()


def test_relu_layer():
    layer = ReLULayer(5, 10)
    btm_data = np.random.randn(5, 10)
    print btm_data
    layer.forward(btm_data)
    print layer.top_data
    top_diff = np.random.randn(5, 10)
    print top_diff
    layer.backward(top_diff)
    print layer.btm_diff


def test_net_init():
    net = Net('../proto/test.prototxt')
    net.init()


if __name__ == "__main__":
    #test_layer()
    #test_inner_product_layer()
    #test_soft_max_layer()
    #test_cross_entropy_layer()
    #test_relu_layer()
    test_net_init()