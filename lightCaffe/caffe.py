__author__ = 'sichen'

import argparse
from lightCaffe.solver import *


def train(solver_path):
    solver = Solver(solver_path)
    solver.solve()


parser = argparse.ArgumentParser(description='This is the main script for training, testing and \
                                fine-tuning the network.')
parser.add_argument('-m', '--mode', default='train', dest='mode', help='Choose the mode. Now only \
                train is available. [default: train]')
parser.add_argument('-s', '--solver', dest='solver_path', help='specify the solver path')

args = parser.parse_args()
if args.mode == 'train':
    print args.solver_path
    train(args.solver_path)