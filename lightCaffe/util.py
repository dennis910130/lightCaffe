__author__ = 'sichen'

import numpy as np
from proto.lightCaffe_pb2 import *
from google.protobuf.text_format import *


def is_near_enough(a, b, epsilon=1e-3):
    return np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b)) < epsilon


def parse_net_from_prototxt(file_path):
    net = NetParameter()
    f = open(file_path, 'rb')
    Merge(f.read(), net)
    return net


def parse_solver_from_prototxt(file_path):
    solver = SolverParameter()
    print file_path
    f = open(file_path, 'rb')
    Merge(f.read(), solver)
    return solver