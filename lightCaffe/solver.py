__author__ = 'sichen'
from lightCaffe.net import *
import time
import os


class Solver:
    def __init__(self, solver_file):
        self.solver_param = parse_solver_from_prototxt(solver_file)
        self.test_iter = self.solver_param.test_iter
        self.test_interval = self.solver_param.test_interval
        self.base_lr = self.solver_param.base_lr
        self.display = self.solver_param.display
        self.max_iter = self.solver_param.max_iter
        self.net = Net(os.path.dirname(solver_file) + '/' + self.solver_param.net)

    def solve(self):
        self.net.init()

        print '... training the model'

        start_time = time.clock()
        iteration = 0
        while iteration < self.max_iter:
            self.net.forward_and_backward()
            self.net.update(self.base_lr)

            iteration += 1
            if iteration % self.display == 0:
                print time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()) + \
                    '  iteration %i, loss %f' % (iteration, self.net.total_loss)
            if iteration % self.test_interval == 0:
                test_loss = 0.
                test_error = 0.
                for i in xrange(self.test_interval):
                    loss, error = self.net.forward_test()
                    test_loss += loss
                    test_error += error
                test_loss /= self.test_interval
                test_error /= self.test_interval
                print time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + \
                    '  iteration %i, test loss %f, test error %f %%' % (iteration, test_loss, test_error * 100)

        end_time = time.clock()
        print 'The training procedure took %.1fs' % (end_time - start_time)