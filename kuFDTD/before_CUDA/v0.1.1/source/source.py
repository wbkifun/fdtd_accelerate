#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : source.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 2. 19. Fri

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the Source class.

===============================================================================
"""

from kufdtd.kufdtd_base import *

def calc_slice_indices(grid_opt, apply_field, point1, point2, \
            step=(None, None, None) ):
    (i1, j1, k1) = point1
    (i2, j2, k2) = point2
    try:
        step_x, step_y, step_z = step
    except TypeError:
        print 'Step option must be a tuple (step_x, step_y, step_z).'

    axis = string2axis(apply_field)
    if (grid_opt == 'efaced' and 'e' in apply_field) \
            or (grid_opt == 'hfaced' and 'h' in apply_field):
        indices = [i1+1, i2+1, j1+1, j2+1, k1+1, k2+1]
        indices[axis*2 + 1] += 1
    elif (grid_opt == 'efaced' and 'h' in apply_field) \
            or (grid_opt == 'hfaced' and 'e' in apply_field):
        indices = [i1, i2+1, j1, j2+1, k1, k2+1]
        indices[axis*2 + 1] -= 1
    
    slice_x = slice(indices[0], indices[1], step_x)
    slice_y = slice(indices[2], indices[3], step_y)
    slice_z = slice(indices[4], indices[5], step_z)
    slice_indices = (slice_x, slice_y, slice_z)
    
    shape = (indices[1] - indices[0], \
            indices[3] - indices[2], \
            indices[5] - indices[4])

    return slice_indices, shape


def calc_slice_indices_mpi(grid_opt, apply_field, point1, point2, \
            number_cells, mpicoord, step=(None, None, None)):
    local_point1 = [0,0,0]
    local_point2 = [0,0,0]

    for axis in [x_axis, y_axis, z_axis]:
        pt1, pt2 = point1[axis], point2[axis]
        N = number_cells[axis]
        coord = mpicoord[axis]
        local_pt1, local_pt1 = 0, N

        if (pt1/N > coord or pt2/N < coord):
            local_pt2 = 0
        else:
            if pt1/N == coord:
                local_pt1 = pt1 - N*coord
            if pt2/N == coord:
                local_pt2 = pt2 - N*coord

        local_point1[axis] = local_pt1
        local_point2[axis] = local_pt2

    return calc_slice_indices(grid_opt, apply_field, \
            local_point1, local_point2, step)


def Source:
    def __init__(self, apply_field, point1, point2, \
            temporal_func, geometrical_func='uniform', \
            apply_type = 'soft', amplitude=1):
        self.apply_field = apply_field
        self.point1 = point1
        self.point2 = point2
        self.geometrical_func = geometrical_func
        self.temporal_func = temporal_func
        self.apply_type = apply_type
        self.amplitude = amplitude


    def discretize(self, ds, grid_opt):
        self.ds = ds
        point1 = real2discrete(ds, self.point1)
        point2 = real2discrete(ds, self.point2)
        self.slice_indices, self.geometrical_shape = \
                calc_slice_indices(\
                ds, grid_opt, self.apply_field, point1, point2)*


    def set_apply_field(self, space):
        axis = string2axis(self.apply_field)
        if 'e' in self.apply_field:
            field = space.efield[axis]
        elif 'h' in self.apply_field:
            field = space.hfield[axis]

        slice_x, slice_y, slice_z = self.slice_indices
        self.source_field = field[slice_x, slice_y, slice_z]

        if self.apply_type == 'soft':
            self.base_field = self.source_field
        elif self.apply_type == 'hard':
            self.base_field = 0


    def set_geometrical_weight(self, \
            number_cells=(0,0,0), mpicoord=(0,0,0)): # for mpi
        if self.geometrical_func[0] == 'uniform':
            weight = 1
        elif self.geometrical_func[0] == 'gaussian':
            self.i0, self.j0, self.sigma_x, self.sigma_y = \
                    real2discrete(self.ds, \
                    self.geometrical_func[1], \
                    self.geometrical_func[2], \
                    self.geometrical_func[3], \
                    self.geometrical_func[4])

            weight = geometrical_gaussian_func(number_cells, mpicoord)

        self.geometrical_weight = self.amplitude*weight


    def geometrical_gaussian_func(self, number_cells, mpicoord):
        list_shape = list(self.geometrical_shape)
        if list_shape.count(1) == 1:
            nonused_axis = list_shape.index( min(list_shape) )

            i_axis, j_axis = [x_axis, y_axis, z_axis].pop(nonused_axis)
            shift_i0 = i0 - self.slice_indices[i_axis].start - \
                    number_cells[i_axis]*mpicoord[i_axis]
            shift_j0 = j0 - self.slice_indices[j_axis].start - \
                    number_cells[j_axis]*mpicoord[j_axis]

            list_shape.pop(nonused_axis)
            nx, ny = list_shape
            array_x = sc.arange(nx).astype('f') - shift_i0
            array_x[:] = sc.exp( -0.5*( (array_x[:]/sigma_x)**2 ) )
            array_y = sc.arange(ny).astype('f') - shift_j0
            array_y[:] = sc.exp( -0.5*( (array_y[:]/sigma_y)**2 ) )
            local_field = array_x[:,sc.newaxis] * array_y

        elif list_shape.count(1) == 2:
            used_axis = list_shape.index( max(list_shape) )

            i_axis = used_axis
            shift_i0 = i0 - self.slice_indices[i_axis].start - \
                    number_cells[i_axis]*mpicoord[i_axis]

            nx = list_shape.pop(used_axis)
            array_x = sc.arange(nx).astype('f') - shift_i0
            local_field = sc.exp( -0.5*( (array_x[:]/sigma_x)**2 ) )

        return local_field


    def set_temporal_function(self, dt):
        self.dt = dt

        if self.temporal_func[0] == 'sin':
            wavelength = self.temporal_func[1]
            self.wfreq = 2*pi*light_velocity/wavelength
            rotate_theta = self.temporal_func[2][0]*pi/180
            rotate_phi = self.temporal_func[2][1]*pi/180
            if rotate_theta == 0 and rotate_phi == 0:
                func = self.temporal_sin_func
            else:
                self.set_rotate_phase(wavelength, rotate_theta, rotate_phi)
                func = self.temporal_sin_phase_func

        elif self.temporal_func[0] == 'gaussian':
            self.t0 = self.temporal_func[1]
            self.sigma_t = self.temporal_func[2]
            wavelength = self.temporal_func[3]
            self.wfreq = 2*pi*light_velocity/wavelength
            func = self.temporal_gaussian_func

        elif self.temporal_func[0] == 'ping':
            self.t0 = self.temporal_func[1]
            func = self.temporal_ping_func

        self.temporal_function = func


    def set_rotate_phase(self, wavelength, theta, phi, \
            number_cells=(0,0,0), mpicoord=(0,0,0)): # for mpi
        ca = 2*pi*self.ds/wavelength

        list_shape = list(self.geometrical_shape)
        if list_shape.count(1) == 1:
            nonused_axis = list_shape.index( min(list_shape) )

            i_axis, j_axis = [x_axis, y_axis, z_axis].pop(nonused_axis)
            shift_i = self.slice_indices[i_axis].start + \
                    number_cells[i_axis]*mpicoord[i_axis]
            shift_j = self.slice_indices[j_axis].start + \
                    number_cells[j_axis]*mpicoord[j_axis]

            list_shape.pop(nonused_axis)
            nx, ny = list_shape
            phase = sc.zeros((nx,ny), 'f')
            for i in xrange(nx):
                for j in xrange(ny):
                    xi = i + shift_i
                    yj = j + shift_j
                    phase[i,j] = ca*sc.sqrt(xi**2 + yj**2 - \
                            (xi*sc.sin(theta)*sc.cos(phi) + \
                            yj*sc.cos(theta))**2)

        elif list_shape.count(1) == 2:
            used_axis = list_shape.index( max(list_shape) )

            i_axis = used_axis
            shift_i0 = i0 - self.slice_indices[i_axis].start - \
                    number_cells[i_axis]*mpicoord[i_axis]

            nx = list_shape.pop(used_axis)
            phase = sc.zeros(nx, 'f')
            for i in xrange(nx):
                xi = i + shift_i
                phase[i] = ca*sc.sqrt(xi**2 + (xi*sc.cos(theta))**2)

        self.geometrical_phase = phase


    def temporal_sin_func(self, tstep):
        return sc.sin(self.wfreq*self.dt*tstep)


    def temporal_sin_phase_func(self, tstep):
        return sc.sin(self.wfreq*self.dt*tstep + self.geometrical_phase)


    def temporal_gaussian_func(self, tstep): 
        return sc.exp( -0.5*( (tstep-self.t0)/self.sigma_t)**2 ) ) \
                *sc.cos( self.wfreq*self.dt*(tstep-self.t0) )


    def temporal_ping_func(self, tstep):
        if self.t0 == tstep:
            return 1
        else:
            return 0


    def update(self, tstep):
        self.source_field = self.base_field + \
                self.geometrical_weight*self.temporal_func(tstep)
