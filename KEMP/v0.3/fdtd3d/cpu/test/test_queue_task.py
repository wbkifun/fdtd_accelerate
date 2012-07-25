import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.cpu import QueueTask


class TestQueueTask(unittest.TestCase):
    def __init__(self, args):
        super(TestQueueTask, self).__init__()
        self.args = args


    def doubling(self, x, y):
        y[:] = 2 * x[:]


    def combine(self, x, y, z):
        z[:] = np.concatenate((x, y))


    def verify(self, x, y):
        self.assertEqual(np.linalg.norm(x - y), 0)


    def runTest(self):
        nx, tmax = self.args

        source = np.random.rand(nx)
        result = np.zeros_like(source)
        self.doubling(source, result)

        arr0 = np.zeros(nx/2)
        arr1 = np.zeros_like(arr0)
        arr2 = np.zeros(nx)

        qtask0 = QueueTask()
        qtask1 = QueueTask()

        for i in xrange(tmax):
            evt0 = qtask0.enqueue(self.doubling, [source[:nx/2], arr0])
            evt1 = qtask1.enqueue(self.doubling, [source[nx/2:], arr1])
            evt2 = qtask0.enqueue(self.combine, [arr0, arr1, arr2], wait_for=[evt1])
            evt3 = qtask1.enqueue(self.verify, [result, arr2], wait_for=[evt2])
        evt3.wait()



if __name__ == '__main__':
    suite = unittest.TestSuite() 
    suite.addTest(TestQueueTask( [1000, 10000] ))

    unittest.TextTestRunner().run(suite) 
