import numpy as np
import unittest

from kemp.fdtd3d.util import common
from kemp.fdtd3d.node import NodeFields
from kemp.fdtd3d import gpu, cpu


class NodeExchange:
    def __init__(self, node_fields):
        """
        """

        common.check_type('node_fields', node_fields, NodeFields)

        # local variables
        nodef = node_fields
        mainf_list = nodef.mainf_list
        cpuf_dict = nodef.cpuf_dict

        self.setf_dict = {'e': [], 'h': []}
        self.getf_dict = {'e': [], 'h': []}
        self.setf_block_dict = {'e': [], 'h': []}
        self.getf_block_dict = {'e': [], 'h': []}

        anx_list = nodef.accum_nx_list

        # main and x+
        for f0, f1 in zip(mainf_list[:-1], mainf_list[1:]):
            self.append_setf_getf_pair('x', f0, f1)

        # x-
        if cpuf_dict.has_key('x-'):
            cf = cpuf_dict['x-']
            gf = mainf_list[0]
            self.append_setf_getf_pair('x', cf, gf, True)

        # y+
        if cpuf_dict.has_key('y+'):
            cf = cpuf_dict['y+']
            for gf, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                self.append_setf_getf_pair('y', gf, cf, True, [anx0, anx1-1])

        # y-
        if cpuf_dict.has_key('y-'):
            cf = cpuf_dict['y-']
            for gf, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                self.append_setf_getf_pair('y', cf, gf, True, [anx0, anx1-1])

        # z+
        if cpuf_dict.has_key('z+'):
            cf = cpuf_dict['z+']
            for gf, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                self.append_setf_getf_pair('z', gf, cf, True, [anx0, anx1-1])

        # z-
        if cpuf_dict.has_key('z-'):
            cf = cpuf_dict['z-']
            for gf, anx0, anx1 in zip(mainf_list, anx_list[:-1], anx_list[1:]):
                self.append_setf_getf_pair('z', cf, gf, True, [anx0, anx1-1])

        # global variables
        self.mainf_list = mainf_list
        self.cpuf_dict = cpuf_dict


    def append_setf_getf_pair(self, axis, f0, f1, is_buffer=False, \
            split_pty=None):
        self.gpu, self.cpu = gpu, cpu
        type0 = f0.device_type
        type1 = f1.device_type
        Set0 = getattr(self, type0).SetFields
        Get0 = getattr(self, type0).GetFields
        Set1 = getattr(self, type1).SetFields
        Get1 = getattr(self, type1).GetFields

        strf_dict = { \
                'x': {'e': ['ey', 'ez'], 'h': ['hy', 'hz']}, \
                'y': {'e': ['ex', 'ez'], 'h': ['hx', 'hz']}, \
                'z': {'e': ['ex', 'ey'], 'h': ['hx', 'hy']} }
        strf = strf_dict[axis]
        axis_id = {'x':0, 'y':1, 'z':2}[axis]
        shift = 1 if is_buffer else 0

        is_yz_buffer = lambda f: True \
                if f.device_type == 'cpu' and \
                f.mpi_type in ['y+', 'y-', 'z+', 'z-'] \
                else False

        strf0 = strf_dict['x'] if is_yz_buffer(f0) else strf
        strf1 = strf_dict['x'] if is_yz_buffer(f1) else strf
        aid0 = 0 if is_yz_buffer(f0) else axis_id
        aid1 = 0 if is_yz_buffer(f1) else axis_id

        # generate the pt0 and pt1
        base0 = lambda f, d: [0+d, split_pty[0]+d, 0+d] \
                if is_yz_buffer(f) and split_pty != None \
                else [0+d, 0+d, 0+d]

        base1 = lambda f, d: [f.nx-2+d, split_pty[1]+d, f.nz-2+d] \
                if is_yz_buffer(f) and split_pty != None \
                else [f.nx-2+d, f.ny-2+d, f.nz-2+d]

        replace = lambda lst, idx, val: lst[:idx] + [val] + lst[idx+1:]

        ns0 = f0.ns[aid0]
        f0_e_pt0 = replace(base0(f0, 0), aid0, ns0-1) 
        f0_e_pt1 = replace(base1(f0, 0), aid0, ns0-1) 
        f1_e_pt0 = replace(base0(f1, 0), aid1, shift) 
        f1_e_pt1 = replace(base1(f1, 0), aid1, shift) 

        f1_h_pt0 = replace(base0(f1, 1), aid1, 0) 
        f1_h_pt1 = replace(base1(f1, 1), aid1, 0) 
        f0_h_pt0 = replace(base0(f0, 1), aid0, ns0-1-shift) 
        f0_h_pt1 = replace(base1(f0, 1), aid0, ns0-1-shift) 

        # create the set and get instances
        set_e = Set0(f0, strf0['e'], f0_e_pt0, f0_e_pt1, True)
        get_e = Get1(f1, strf1['e'], f1_e_pt0, f1_e_pt1)
        set_h = Set1(f1, strf1['h'], f1_h_pt0, f1_h_pt1, True)
        get_h = Get0(f0, strf0['h'], f0_h_pt0, f0_h_pt1)

        # global variables
        if type0 == 'gpu' and type1 == 'cpu': 
            self.setf_block_dict['e'].append(set_e)
            self.getf_block_dict['e'].append(get_e)
        else:
            self.setf_dict['e'].append(set_e)
            self.getf_dict['e'].append(get_e)

        if type0 == 'cpu' and type1 == 'gpu': 
            self.setf_block_dict['h'].append(set_h)
            self.getf_block_dict['h'].append(get_h)
        else:
            self.setf_dict['h'].append(set_h)
            self.getf_dict['h'].append(get_h)


    def update(self, str_eh):
        setf_list = self.setf_dict[str_eh]
        getf_list = self.getf_dict[str_eh]
        for setf, getf in zip(setf_list, getf_list):
            setf.set_fields(getf.get_fields(), [getf.get_event()])

        setf_list = self.setf_block_dict[str_eh]
        getf_list = self.getf_block_dict[str_eh]
        for setf, getf in zip(setf_list, getf_list):
            getf.get_event().wait()
            setf.set_fields( getf.get_fields() )


    def update_e(self):
        self.update('e')


    def update_h(self):
        self.update('h')
