__author__ = 'sichen'
from lightCaffe.layer import *
from lightCaffe.util import *


class Net:
    def __init__(self, param_file):
        self.net_param = parse_net_from_prototxt(param_file)
        self.n_layer = len(self.net_param.layer)
        self.layers = []

    def print_proto_file(self):
        print self.net_param

    def init(self):
        print '... initializing network from parameters:'
        self.print_proto_file()
        self._init_data_layer()
        self.layers[0].print_information()
        for i in range(1, self.n_layer):
            self._append(self.net_param.layer[i])
            self.layers[-1].print_information()

    def _init_data_layer(self):
        data_layer_param = self.net_param.layer[0]
        if data_layer_param.type == 'pkl_data_layer':
            data_layer = PklDataLayer(layer_param=data_layer_param)
        self.layers.append(data_layer)
        data_layer.load_data()

    def _append(self, layer_param):
        if layer_param.type == 'inner_product_layer':
            layer = InnerProductLayer(self.layers[0].n_batch, self.layers[-1].n_out, layer_param=layer_param)
        elif layer_param.type == 'soft_max_layer':
            layer = SoftMaxLayer(self.layers[0].n_batch, self.layers[-1].n_out, layer_param=layer_param)
        elif layer_param.type == 'cross_entropy_layer':
            layer = CrossEntropyLossLayer(self.layers[0].n_batch, self.layers[-1].n_out, layer_param=layer_param)
        self.layers.append(layer)

    def forward(self):
        (data_input, label) = self.layers[0].get_next_batch_train()
        for i in range(1, self.n_layer-1):
            self.layers[i].forward(data_input)
            data_input = self.layers[i].top_data
        self.layers[-1].forward(data_input, label)

    def backward(self):
        self.layers[-1].backward()
        for i in range(self.n_layer-1, 0, -1):
            self.layers[i].backward(self.layers[i+1].btm_diff)

    def update(self, learning_rate):
        for i in range(0, self.n_layer):
            if self.layers[i].need_update:
                self.layers[i].update(learning_rate)

    def forward_val(self):
        (data_input, label) = self.layers[0].get_next_batch_val()
        for i in range(1, self.n_layer-1):
            self.layers[i].forward(data_input)
            data_input = self.layers[i].top_data
        self.layers[-1].forward(data_input, label)
        return self.layers[-1].total_loss, self.layers[-1].error()