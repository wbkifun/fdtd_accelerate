import atexit
import numpy as np
import unittest
from ctypes import c_int
from threading import Thread, Event
from Queue import Queue

from kemp.fdtd3d.util import common, common_cpu
from kemp.fdtd3d.test import common_update


class QueueTask(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.queue = Queue()
        atexit.register( self.queue.join )

        self.start()


    def run(self):
        while True:
            func, args, wait_for, event = self.queue.get()

            for evt in wait_for: 
                evt.wait()
            func(*args)
            event.set()

            self.queue.task_done()


    def enqueue(self, func, args=[], wait_for=[]):
        event = Event()
        event.clear()
        self.queue.put( (func, args, wait_for, event) )

        return event 


    def enqueue_barrier(self):
        evt = self.enqueue(lambda:None)
        evt.wait()



class Fields:
    def __init__(self, nx, ny, nz, \
            coeff_use='e', \
            precision_float='single', \
            use_cpu_core=0):
        """
        """

        common.check_type('nx', nx, int)
        common.check_type('ny', ny, int)
        common.check_type('nz', nz, int)
        common.check_value('coeff_use', coeff_use, ('', 'e', 'h', 'eh'))
        common.check_value('precision_float', precision_float, ('single', 'double'))

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.coeff_use=coeff_use
        self.dtype = {'single':np.float32, 'double':np.float64}[precision_float]
        self.dtype_str_list = { \
                'single':['float', 'xmmintrin.h', 'ps', '__m128', '4', '0, 1, 1, 1'], \
                'double':['double', 'emmintrin.h', 'pd', '__m128d', '2', '0, 1'] }[precision_float]

        self.device_type = 'cpu'

        # padding for the nz which is multiple of 4 (float32) or 2 (float64)
        a_size = {'single':4, 'double':2}[precision_float]   # 16 Bytes
        self.pad = pad = int(np.ceil(float(nz) / a_size) * a_size) - nz
        self.slz = slice(None, None) if pad == 0 else slice(None, -pad)
        self.nz_pitch = nz_pitch = nz + pad

        mask_arr = np.ones(a_size, 'i')
        mask_arr[-(pad+1):] = 0
        self.dtype_str_list.append( str(list(mask_arr)).strip('[]') )

        # ns, qtask, enqueue
        self.ns = [nx, ny, nz]
        self.ns_pitch = [nx, ny, nz_pitch]
        self.qtask = QueueTask()
        self.enqueue = self.qtask.enqueue
        self.enqueue_barrier = self.qtask.enqueue_barrier

        # on/off the coefficient arrays
        self.ce_on = True if 'e' in self.coeff_use else False
        self.ch_on = True if 'h' in self.coeff_use else False

        # allocations
        self.ehs = [np.zeros(self.ns_pitch, dtype=self.dtype) for i in range(6)]
        self.ex, self.ey, self.ez, self.hx, self.hy, self.hz = self.ehs

        if self.ce_on:
            self.ces = [np.ones(self.ns_pitch, dtype=self.dtype)*0.5 for i in range(3)]
            self.cex, self.cey, self.cez = self.ces 

        if self.ch_on:
            self.chs = [np.ones(self.ns_pitch, dtype=self.dtype)*0.5 for i in range(3)]
            self.chx, self.chy, self.chz = self.chs

        # program
        macros = [ \
                'ARGS_CE', 'INIT_CE', 'PRIVATE_CE', 'CEX', 'CEY', 'CEZ', \
                'ARGS_CH', 'INIT_CH', 'PRIVATE_CH', 'CHX', 'CHY', 'CHZ', \
                'OMP_SET_NUM_THREADS', \
                'DTYPE', 'MM_HEADER', 'PSD', 'TYPE128', 'INCRE', 'MASK_H', 'MASK_E']

        values = [ \
                '', 'ce=SET1(0.5)', '', '', '', '', \
                '', 'ch=SET1(0.5)', '', '', '', '', \
                ''] + self.dtype_str_list

        if use_cpu_core != 0:
            values[12] = 'omp_set_num_threads(%d);' % use_cpu_core

        if self.ce_on:
            values[:6] = [ \
                    ', DTYPE *cex, DTYPE *cey, DTYPE *cez', 'ce', ', ce', \
                    'ce = LOAD(cex+idx);', 'ce = LOAD(cey+idx);', 'ce = LOAD(cez+idx);']
        if self.ch_on:
            values[6:12] = [ \
                    ', DTYPE *chx, DTYPE *chy, DTYPE *chz', 'ch', ', ch', \
                    'ch = LOAD(chx+idx);', 'ch = LOAD(chy+idx);', 'ch = LOAD(chz+idx);']

        ksrc = common.replace_template_code( \
                open(common_cpu.src_path + 'core.c').read(), macros, values)
        self.program = common_cpu.build_clib(ksrc)

        carg = np.ctypeslib.ndpointer(dtype=self.dtype, ndim=3, \
                shape=(nx, ny, nz_pitch), flags='C_CONTIGUOUS, ALIGNED')
        argtypes = [c_int, c_int, c_int, c_int, c_int] + \
                [carg for i in xrange(6)]
        self.program.update_e.argtypes = argtypes
        self.program.update_e.restype = None
        self.program.update_h.argtypes = argtypes
        self.program.update_h.restype = None

        self.e_args = self.ns_pitch + [0, nx*ny*nz_pitch] + self.ehs
        self.h_args = self.ns_pitch + [0, nx*ny*nz_pitch] + self.ehs

        if self.ce_on:
            self.program.update_e.argtypes += [carg for i in xrange(3)]
            self.e_args += self.ces

        if self.ch_on:
            self.program.update_h.argtypes += [carg for i in xrange(3)]
            self.h_args += self.chs


    def get(self, str_f):
        return self.__dict__[str_f][:,:,self.slz]


    def set_ces(self, cex, cey, cez):
        if self.ce_on:
            self.cex[:,:,self.slz] = cex[:]
            self.cey[:,:,self.slz] = cey[:]
            self.cez[:,:,self.slz] = cez[:]
        else:
            raise AttributeError("The Fields instance has no ce arrays. You should add 'e' in the option 'coeff_use'")


    def set_chs(self, chx, chy, chz):
        if self.ch_on:
            self.chx[:,:,self.slz] = chx[:]
            self.chy[:,:,self.slz] = chy[:]
            self.chz[:,:,self.slz] = chz[:]
        else:
            raise AttributeError("The Fields instance has no ch arrays. You should add 'h' in the option 'coeff_use'")


    def update_e(self):
        self.qtask.enqueue(self.program.update_e, self.e_args)


    def update_h(self):
        self.qtask.enqueue(self.program.update_h, self.h_args)




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



class TestFields(unittest.TestCase):
    def __init__(self, args):
        super(TestFields, self).__init__()
        self.args = args


    def runTest(self):
        ufunc, nx, ny, nz, coeff_use, precision_float, use_cpu_core, tmax = self.args
        fields = Fields(nx, ny, nz, coeff_use, precision_float, use_cpu_core)
        strf_list = ['ex', 'ey', 'ez', 'hx', 'hy', 'hz']

        # allocations
        ns = fields.ns
        dtype = fields.dtype

        ehs = common_update.generate_random_ehs(ufunc, nx, ny, nz, dtype)
        for strf, eh in zip(strf_list, ehs):
            fields.get(strf)[:] = eh

        ces, chs = common_update.generate_random_cs(coeff_use, nx, ny, nz, dtype)
        if 'e' in coeff_use:
            fields.set_ces(*ces)
        if 'h' in coeff_use:
            fields.set_chs(*chs)

        # update
        if ufunc == 'e':
            for tstep in xrange(0, tmax):
                fields.update_e()
                common_update.update_e(ehs, ces)
            fields.enqueue_barrier()

            for strf, eh in zip(strf_list, ehs):
                norm = np.linalg.norm(eh - fields.get(strf))
                max_diff = np.abs(eh - fields.get(strf)).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % \
                        (self.args, strf, norm, max_diff) )

        elif ufunc == 'h':
            for tstep in xrange(0, tmax):
                fields.update_h()
                common_update.update_h(ehs, chs)
            fields.enqueue_barrier()

            for strf, eh in zip(strf_list, ehs):
                norm = np.linalg.norm(eh - fields.get(strf))
                max_diff = np.abs(eh - fields.get(strf)).max()
                self.assertEqual(norm, 0, '%s, %s, %g, %g' % \
                        (self.args, strf, norm, max_diff) )



if __name__ == '__main__':
    suite = unittest.TestSuite() 
    suite.addTest(TestQueueTask( [1000, 10000] ))

    nx, ny = 40, 50
    #suite.addTest(TestFields( ('e', nx, ny, 64, '', 'single', 0, 1) ))

    args_list = [ \
            (ufunc, nx, ny, nz, coeff_use, precision_float, use_cpu_core, 1) \
            for ufunc in ['e', 'h'] \
            for nz in [61, 62, 63, 64] \
            for coeff_use in ['', 'e', 'h'] \
            for precision_float in ['single', 'double'] \
            for use_cpu_core in [0, 1] ]
    suite.addTests(TestFields(args) for args in args_list) 

    fields = Fields(nx, ny, nz, coeff_use, precision_float, use_cpu_core)
    suite.addTest(TestFields( ('e', nx, ny, 64, 'e', 'single', 0, 10) ))
    suite.addTest(TestFields( ('e', nx, ny, 64, 'e', 'double', 0, 10) ))

    unittest.TextTestRunner().run(suite) 
