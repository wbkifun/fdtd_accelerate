#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : drude.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 2. 15. Fri

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the class for the matter of Drude type.

===============================================================================
"""

from matter_base import *
from core.drude_core import update_drude
from core.non_dispersive_core import update_non_dispersive

class Drude(MatterBase):
    """
    The drude type matter class
    """
    def set_drude_parameters(self, epr_infinity, plasma_freq, gamma):
        self.epr_inf_val = epr_infinity
        self.pfreq_val = plasma_freq
        self.gamma_val = gamma


    def allocate_arrays(self):
        self.efield = alloc_numpy_arrays(self.e_number_cells, 'xyz')
        self.hfield = alloc_numpy_arrays(self.h_number_cells, 'xyz')
        self.ffield = alloc_numpy_arrays(self.e_number_cells, 'xyz')

        self.cea = alloc_numpy_arrays(self.e_number_cells, 'xyz', 1)
        tmp_ceb = 2*self.dt/(ep0*self.ds)*self.unit_factor
        self.ceb = alloc_numpy_arrays(self.e_number_cells, 'xyz', tmp_ceb)
        self.cef = alloc_numpy_arrays(self.e_number_cells, 'xyz')

        tmp_chb = self.dt/(mu0*self.ds)/self.unit_factor
        self.chb = alloc_numpy_arrays((1,1,1), 'xyz', tmp_chb)


    def set_coefficients(self):
        epr_inf = alloc_numpy_arrays(self.e_number_cells, 'xyz', 1)
        pfreq = alloc_numpy_arrays(self.e_number_cells, 'xyz')
        self.gamma = alloc_numpy_arrays(self.e_number_cells, 'xyz')

        pfreq2 = calc_with_list(pfreq, '**', 2)

        for axis in [x_axis, y_axis, z_axis]:
            self.cea[axis][:,:,:] = \
                    1./( 2 + pfreq2[axis][:,:,:]*(self.dt**2) ) 
            self.ceb[axis][:,:,:] = \
                    ( self.cea[axis][:,:,:]/epr_inf[axis][:,:,:] ) \
                    *self.ceb[axis][:,:,:]
            self.cef[axis][:,:,:] = \
                    self.cea[axis][:,:,:] \
                    *pfreq2[axis][:,:,:] \
                    *self.dt \
                    *( sc.exp(-self.gamma[axis][:,:,:]*self.dt) + 1 )

        free_numpy_arrays(epr_inf)
        free_numpy_arrays(pfreq)


    def update_e(self):
        if self.grid_opt == 'efaced':
            in_out_field = "out_field"
        elif self.grid_opt == 'hfaced':
            in_out_field = "in_field"
        update_drude(
                self.grid_opt,
                in_out_field,
                self.dt,
                self.number_cells,
                self.efield, # update_field 
                self.hfield, # base_field
                self.ffield,
                self.gamma,
                self.cea,
                self.ceb,
                self.cef)

    def update_h(self):
        if self.grid_opt == 'efaced':
            in_out_field = 'in_field'
        elif self.grid_opt == 'hfaced':
            in_out_field = 'out_field'
        update_non_dispersive(
                self.grid_opt,
                in_out_field,
                self.number_cells,
                self.hfield, # update_field
                self.efield, # base_field
                self.chb)


#==============================================================================
# test code
#==============================================================================
if __name__ == '__main__':        
    from time import *
    from scipy import sin, exp, sqrt

    ds = 10e-9
    number_cells = [200, 200, 30]
    grid_opt = 'efaced'
    unit_opt = 'Enorm'

    # construct the matter object
    space = Drude(ds, number_cells, grid_opt, unit_opt)
    space.set_drude_parameters(
            epr_infinity = 9.0685, 
            plasma_freq = 2*pi*2155.6e12/sqrt(9.0685), 
            gamma = 2*pi*18.36e12)	# modified P.R.B parameters
    space.allocate_arrays()
    Nx, Ny, Nz = number_cells[0], number_cells[1], number_cells[2]
    Ex, Ey, Ez = space.efield[0], space.efield[1], space.efield[2]
    space.set_coefficients()

    # for graphics using matplotlib
    from pylab import *
    ion()
    figure(figsize=(10,5))

    # for sin source
    wavelength = 300e-9
    wfreq = light_velocity*2*pi/wavelength

    # for data capture
    cap_t = 10 # capture_time
    cap_pt = Nx/2 # capture_point

    #==========================================================================
    # main time loop
    #==========================================================================
    t0 = time()
    for tstep in xrange(100000):
        space.update_e()

        pulse = sin(wfreq*space.dt*tstep)
        space.efield[2][Nx/3, Ny/3*2, :] += pulse

        space.update_h()

        if tstep/cap_t*cap_t == tstep:
            t1 = time()
            elapse_time = localtime(t1-t0-60*60*9)
            str_time = strftime('[%j]%H:%M:%S', elapse_time)
            print '%s    tstep = %d' % (str_time, tstep)

            intensity = (Ex[:,:,14]**2 + Ey[:,:,14] + Ez[:,:,14]**2)
            #clf
            imshow(rot90(intensity),
                    cmap=cm.hot,
                    vmin=0, vmax=0.001,
                    interpolation='bilinear')
            #colorbar()
            draw()
