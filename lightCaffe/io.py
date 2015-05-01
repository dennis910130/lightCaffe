__author__ = 'sichen'
from proto.lightCaffe_pb2 import *
from google.protobuf.text_format import *


def parse_net_from_prototxt(file_path):
    net = NetParameter()
    f = open(file_path, 'rb')
    Merge(f.read(), net)
    return net