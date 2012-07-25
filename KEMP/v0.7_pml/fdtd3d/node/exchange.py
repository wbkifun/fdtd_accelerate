import numpy as np
import unittest

from kemp.fdtd3d.util import common, common_exchange
from kemp.fdtd3d import cpu

from fields import Fields


class ExchangeNode:
    def __init__(self, node_fields):
        """
        """

        common.check_type('node_fields', node_fields, Fields)

        # local variables
        nodef = node_fields
        mainf_list = nodef.mainf_list
        buffer_dict = nodef.buffer_dict
        anx_list = nodef.accum_nx_list

        # global variables
        self.cpu = cpu
        if 'gpu' in [f.device_type for f in nodef.updatef_list]:
            from kemp.fdtd3d import gpu
            self.gpu = gpu

        self.getf_dict = {'e': [], 'h': []}
        self.setf_dict = {'e': [], 'h': []}
        self.getf_block_dict = {'e': [], 'h': []}
        self.setf_block_dict = {'e': [], 'h': []}

        # main and x buffers
        f_list = mainf_list[:]
        if buffer_dict.has_key('x-'):
            f_list.insert(0, buffer_dict['x-'])
        if buffer_dict.has_key('x+'):
            f_list.append(buffer_dict['x+'])

        f0_list = f_list[:-1]
        f1_list = f_list[1:]
        for f0, f1 in zip(f0_list, f1_list):
            for e_or_h in ['e', 'h']:
                self.put_x(e_or_h, f0, f1)

        # y+ buffer
        if buffer_dict.has_key('y+'):
            f1 = buffer_dict['y+']
            for f0, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                for e_or_h in ['e', 'h']:
                    self.put_yz(e_or_h, f0, f1, 'y', anx0, anx1-1)

        # y- buffer
        if buffer_dict.has_key('y-'):
            f0 = buffer_dict['y-']
            for f1, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                for e_or_h in ['e', 'h']:
                    self.put_yz(e_or_h, f0, f1, 'y', anx0, anx1-1)

        # z+ buffer
        if buffer_dict.has_key('z+'):
            f1 = buffer_dict['z+']
            for f0, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                for e_or_h in ['e', 'h']:
                    self.put_yz(e_or_h, f0, f1, 'z', anx0, anx1-1)

        # z- buffer
        if buffer_dict.has_key('z-'):
            f0 = buffer_dict['z-']
            for f1, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                for e_or_h in ['e', 'h']:
                    self.put_yz(e_or_h, f0, f1, 'z', anx0, anx1-1)

        # append to the update list
        self.priority_type = 'exchange'
        nodef.append_instance(self)


    def put_x(self, e_or_h, f0, f1):
        gf, sf = (f1, f0) if e_or_h == 'e' else (f0, f1)
        strfs = common_exchange.str_fs_dict['x'][e_or_h]

        if isinstance(f0, cpu.BufferFields) or isinstance(f1, cpu.BufferFields):
            gpt0 = common_exchange.pt0_buf_dict(*gf.ns)['x'][e_or_h]['get']
            gpt1 = common_exchange.pt1_buf_dict(*gf.ns)['x'][e_or_h]['get']
        else:
            gpt0 = common_exchange.pt0_dict(*gf.ns)['x'][e_or_h]['get']
            gpt1 = common_exchange.pt1_dict(*gf.ns)['x'][e_or_h]['get']

        spt0 = common_exchange.pt0_dict(*sf.ns)['x'][e_or_h]['set']
        spt1 = common_exchange.pt1_dict(*sf.ns)['x'][e_or_h]['set']

        self.put_getf_setf_list(e_or_h, gf, strfs, gpt0, gpt1, sf, strfs, spt0, spt1)


    def put_yz(self, e_or_h, f0, f1, axis, anx0, anx1):
        is_buf0 = True if isinstance(f0, cpu.BufferFields) else False
        is_buf1 = True if isinstance(f1, cpu.BufferFields) else False
        gf, sf = (f1, f0) if e_or_h == 'e' else (f0, f1)

        if is_buf0:
            ax0, ax1 = 'x', axis
        elif is_buf1:
            ax0, ax1 = axis, 'x'
        gax, sax = (ax1, ax0) if e_or_h == 'e' else (ax0, ax1)

        gstr = common_exchange.str_fs_dict[gax][e_or_h]
        sstr = common_exchange.str_fs_dict[sax][e_or_h]

        gpt0 = common_exchange.pt0_buf_dict(*gf.ns)[gax][e_or_h]['get']
        gpt1 = common_exchange.pt1_buf_dict(*gf.ns)[gax][e_or_h]['get']
        spt0 = common_exchange.pt0_dict(*sf.ns)[sax][e_or_h]['set']
        spt1 = common_exchange.pt1_dict(*sf.ns)[sax][e_or_h]['set']

        if (is_buf0 and e_or_h=='e') or (is_buf1 and e_or_h=='h'):   # sf
            spt0 = (spt0[0], anx0, spt0[2])
            spt1 = (spt1[0], anx1, spt1[2])
        else:
            gpt0 = (gpt0[0], anx0, gpt0[2])
            gpt1 = (gpt1[0], anx1, gpt1[2])

        self.put_getf_setf_list(e_or_h, gf, gstr, gpt0, gpt1, sf, sstr, spt0, spt1)

        '''
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank==0:
            print 'e_or_h', e_or_h
            print 'gf', gf
            print 'gstr', gstr
            print 'gpt0', gpt0
            print 'gpt1', gpt1
            print 'sf', sf
            print 'sstr', sstr
            print 'spt0', spt0
            print 'spt1', spt1
        '''


    def put_getf_setf_list(self, e_or_h, gf, gstr, gpt0, gpt1, sf, sstr, spt0, spt1):
        gtype = getattr(self, gf.device_type)
        stype = getattr(self, sf.device_type)
        getf = gtype.GetFields(gf, gstr, gpt0, gpt1)
        setf = stype.SetFields(sf, sstr, spt0, spt1, True)

        if gtype == 'gpu' and stype == 'cpu':
            self.getf_dict[e_or_h].append(getf)
            self.setf_dict[e_or_h].append(setf)
        else:
            self.getf_block_dict[e_or_h].append(getf)
            self.setf_block_dict[e_or_h].append(setf)


    def update(self, e_or_h):
        setf_list = self.setf_dict[e_or_h]
        getf_list = self.getf_dict[e_or_h]
        for setf, getf in zip(setf_list, getf_list):
            setf.set_fields(getf.get_fields(), [getf.get_event()])

        setf_list = self.setf_block_dict[e_or_h]
        getf_list = self.getf_block_dict[e_or_h]
        for setf, getf in zip(setf_list, getf_list):
            getf.get_event().wait()
            setf.set_fields( getf.get_fields() )


    def update_e(self):
        self.update('e')


    def update_h(self):
        self.update('h')
