import numpy as np
import unittest
from mpi4py import MPI

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common_update
from kemp.fdtd3d.cpu import ExchangeMpi, Fields, GetFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class TestExchangeMpi(unittest.TestCase):
    def test(self):
        nx, ny, nz = 40, 50, 60
        tmax = 10

        # buffer instance
        if rank == 0:
            fields = Fields(10, ny, nz, mpi_type='x+')
            exmpi = ExchangeMpi(fields, 1, tmax)

        elif rank == 1:
            fields = Fields(3, ny, nz, mpi_type='x-')
            exmpi = ExchangeMpi(fields, 0, tmax)

        # generate random source
        nx, ny, nz = fields.ns
        ehs = common_update.generate_random_ehs(nx, ny, nz, fields.dtype)
        fields.set_ehs(*ehs)

        # verify
        for tstep in xrange(1, tmax+1):
            fields.update_e()
            fields.update_h()

        getf_dict = {}
        if rank == 0:
            getf_dict['e'] = GetFields(fields, ['ey', 'ez'], \
                    (nx-1, 0, 0), (nx-1, ny-2, nz-2))

            getf_dict['h'] = GetFields(fields, ['hy', 'hz'], \
                    (1, 1, 1), (1, ny-1, nz-1))

            for eh in ['e', 'h']:
                getf = getf_dict[eh]
                getf.get_event().wait()
                g0 = getf.get_fields()
                g1 = np.zeros_like(g0)
                comm.Recv(g1, 1, tag=10)
                norm = np.linalg.norm(g0 - g1)
                self.assertEqual(norm, 0, '%g, %s, %s' % (norm, 'x', 'e') )

        elif rank == 1:
            getf_dict['e'] = GetFields(fields, ['ey', 'ez'], \
                    (nx-2, 0, 0), (nx-2, ny-2, nz-2))

            getf_dict['h'] = GetFields(fields, ['hy', 'hz'], \
                    (0, 1, 1), (0, ny-1, nz-1))

            for eh in ['e', 'h']:
                getf = getf_dict[eh]
                getf.get_event().wait()
                comm.Send(getf.get_fields(), 0, tag=10)



if __name__ == '__main__':
    unittest.main()
