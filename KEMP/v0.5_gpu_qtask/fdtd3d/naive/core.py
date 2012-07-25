import numpy as np

from kemp.fdtd3d.util import common
from fields import Fields


class Core:
    def __init__(self, fields):
        common.check_type('fields', fields, Fields)

        # global variables
        self.mainf = fields

        pad = fields.pad
        slice_z0 = slice(1, None) if pad == 0 else slice(1, -pad)
        slice_z1 = slice(None, -pad-1)
        self.slice_z_list = [fields.slice_z, slice_z0, slice_z1]

        # append to the update list
        self.priority_type = 'core'
        fields.append_instance(self)


    def update_e(self):
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        cex, cey, cez = self.mainf.ces
        slz, slz0, slz1 = self.slice_z_list

        ex[:,:-1,slz1] += cex[:,:-1,slz1] * \
                ((hz[:,1:,slz1] - hz[:,:-1,slz1]) - (hy[:,:-1,slz0] - hy[:,:-1,slz1]))
        ey[:-1,:,slz1] += cey[:-1,:,slz1] * \
                ((hx[:-1,:,slz0] - hx[:-1,:,slz1]) - (hz[1:,:,slz1] - hz[:-1,:,slz1]))
        ez[:-1,:-1,slz] += cez[:-1,:-1,slz] * \
                ((hy[1:,:-1,slz] - hy[:-1,:-1,slz]) - (hx[:-1,1:,slz] - hx[:-1,:-1,slz]))


    def update_h(self):
        ex, ey, ez, hx, hy, hz = self.mainf.ehs
        chx, chy, chz = self.mainf.chs
        slz, slz0, slz1 = self.slice_z_list

        hx[:,1:,slz0] -= chx[:,1:,slz0] * \
                ((ez[:,1:,slz0] - ez[:,:-1,slz0]) - (ey[:,1:,slz0] - ey[:,1:,slz1]))
        hy[1:,:,slz0] -= chy[1:,:,slz0] * \
                ((ex[1:,:,slz0] - ex[1:,:,slz1]) - (ez[1:,:,slz0] - ez[:-1,:,slz0]))
        hz[1:,1:,slz] -= chz[1:,1:,slz] * \
                ((ey[1:,1:,slz] - ey[:-1,1:,slz]) - (ex[1:,1:,slz] - ex[1:,:-1,slz]))
