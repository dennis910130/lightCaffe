__author__ = 'sichen'
from lightCaffe.layer import *
from lightCaffe.net import *
from lightCaffe.math_util import *

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
    net = Net('../example/logistic_regression_net.prototxt')
    net.init()


def test_im2col():
    im = np.random.randn(3, 10, 10)
    patches = im2col(im, 3, 1)
    print patches.shape
    print patches
    print im[:, 0:3, 0:3]
    print patches[0, :]


def test_im2col_batch():
    im = np.random.randn(2, 3, 10, 10)
    patches = im2col_batch(im, 3, 1)
    print patches.shape
    print patches
    print im[0, :, 0:3, 1:4]
    print patches[2, :]


def test_convolve3d():
    im = np.random.randn(3, 10, 10)
    filter = np.random.randn(2, 3, 2, 2)
    im_out = convolve3d(im, filter, 1)
    print im_out
    print im_out.shape
    print im[:, 8:10, 1:3]
    print filter[1, :]
    print np.sum(im[:, 8:10, 1:3] * filter[0, :])
    print im_out[0, -1, 1]


def test_conv_layer():
    im = np.random.randn(2, 3, 10, 10)
    layer = ConvLayer(2, 3, 10, 10, 0, 2, 2)
    layer.forward(im)
    print layer.top_data.shape
    print im[0, :, 0:2, 1:3]
    print layer.reshaped_batch_data[2]
    print layer.reshaped_W[0]
    print layer.W[0, :]
    print layer.reshaped_out[2]
    print np.sum(im[1, :, 3:5, 1:3] * layer.W[1, :]) + layer.b[1]
    print layer.top_data[1, 1, 3, 1]


def test_col2im_batch():
    im = np.random.randn(2, 3, 10, 10)
    re_im = col2im_batch(im2col_batch(im, 3, 1), 3, 1, 10, 2, 3)
    print im - re_im

if __name__ == "__main__":
    #test_layer()
    #test_inner_product_layer()
    #test_soft_max_layer()
    #test_cross_entropy_layer()
    #test_relu_layer()
    #test_net_init()
    #test_im2col()
    #test_convolve3d()
    #test_conv_layer()
    #test_im2col_batch()
    test_col2im_batch()