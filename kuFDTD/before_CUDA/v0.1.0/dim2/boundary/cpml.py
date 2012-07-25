#!//usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : cpml.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 2. 1. Fri

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the class for the CPML(Convolution Perfect Matched Layer).

===============================================================================
"""

from kufdtd.kufdtd_base import *
from core.cpml_core import update_cpml

def make_gradation_array(number_pml_cells, interval):
    "make a array as symmertry, depth gradation for pml"
    Np = number_pml_cells
    if interval == 'half_depth':
        arr = sc.arange(Np-0.5,-Np-0.5,-1,'f')
        arr[-Np:] = abs(arr[-Np:])
    elif interval == 'full_depth':
        arr = sc.arange(Np,-Np,-1,'f')
        arr[-Np:] = abs(arr[-Np:])+1
    return arr


def alloc_psi_arrays(number_pml_cells, in_out_field):
    Np = number_pml_cells
    if in_out_field == 'in_field':
        psi_yx = sc.zeros((2*Np, Ny+1, Nz+1), 'f')
        psi_zx = sc.zeros((2*Np, Ny+1, Nz+1), 'f')
        psi_zy = sc.zeros((Nx+1, 2*Np, Nz+1), 'f')
        psi_xy = sc.zeros((Nx+1, 2*Np, Nz+1), 'f')
        psi_xz = sc.zeros((Nx+1, Ny+1, 2*Np), 'f')
        psi_yz = sc.zeros((Nx+1, Ny+1, 2*Np), 'f')
    elif in_out_field == 'out_field':
        psi_yx = sc.zeros((2*Np, Ny+2, Nz+2), 'f')
        psi_zx = sc.zeros((2*Np, Ny+2, Nz+2), 'f')
        psi_zy = sc.zeros((Nx+2, 2*Np, Nz+2), 'f')
        psi_xy = sc.zeros((Nx+2, 2*Np, Nz+2), 'f')
        psi_xz = sc.zeros((Nx+2, Ny+2, 2*Np), 'f')
        psi_yz = sc.zeros((Nx+2, Ny+2, 2*Np), 'f')
    return [psi_yx, psi_zx, psi_zy, psi_xy, psi_xz, psi_yz] 


class Cpml:
    """
    The cpml class
    """
    def __init__(self, space, number_pml_cells, cpml_parameter):
        self.space = space
        self.number_pml_cells = number_pml_cells
        self.kapa_max = cpml_parameter[0]
        self.alpha = cpml_parameter[1]
        self.grade_order = cpml_parameter[2]
        self.sigma_max = (self.grade_order+1)/(150*pi*self.space.ds)


    def allocate_pml_arrays(self):
        Np = self.number_pml_cells
        m = self.grade_order
        ds = self.space.ds
        dt = self.space.dt

        if self.space.grid_opt == 'efaced':
            sigma_e = make_gradation_array(Np, 'half_depth')
            sigma_h = make_gradation_array(Np, 'full_depth')
            kapa_e = make_gradation_array(Np, 'half_depth')
            kapa_h = make_gradation_array(Np, 'full_depth')
            self.psi_e = alloc_psi_arrays(Np, 'out_field')        
            self.psi_h = alloc_psi_arrays(Np, 'in_field')        
        elif self.space.grid_opt == 'hfaced':
            sigma_e = make_gradation_array(Np, 'full_depth')
            sigma_h = make_gradation_array(Np, 'half_depth')
            kapa_e = make_gradation_array(Np, 'full_depth')
            kapa_h = make_gradation_array(Np, 'half_depth')
            self.psi_e = alloc_psi_arrays(Np, 'in_field')        
            self.psi_h = alloc_psi_arrays(Np, 'out_field')        

        sigma_e[:] = pow(sigma_e[:]/Np, m)*self.sigma_max
        sigma_h[:] = pow(sigma_h[:]/Np, m)*self.sigma_max 
        kapa_e[:] = 1 + (self.kapa_max - 1)*pow(kapa_e[:]/Np, m)
        kapa_h[:] = 1 + (self.kapa_max - 1)*pow(kapa_h[:]/Np, m)
        rcm_b_e = sc.exp( -(sigma_e[:]/kapa_e[:] + self.alpha)*dt/ep0 )
        mrcm_a_e = ( sigma_e[:]/(sigma_e[:]*kapa_e[:] + 
                self.alpha*(kapa_e[:]**2))*(rcm_b_e[:] - 1) )/ds
        rcm_b_h = sc.exp( -(sigma_h[:]/kapa_h[:] + self.alpha)*dt/ep0 )
        mrcm_a_h = ( sigma_h[:]/(sigma_h[:]*kapa_h[:] +
                self.alpha*(kapa_h[:]**2))*(rcm_b_h[:] - 1))/ds \
                /self.space.unit_factor

        self.pml_coefficient_e = [kapa_e, rcm_b_e, mrcm_a_e]        
        self.pml_coefficient_h = [kapa_h, rcm_b_h, mrcm_a_h]        

        
    def update_cpml_e(self, pml_apply_opt):
        if self.space.grid_opt == 'efaced':
            in_out_field = "out_field"
        elif self.space.grid_opt == 'hfaced':
            in_out_field = "in_field"

        for pml_direction, position in enumerate(pml_apply_opt):    
            for fb in position:
                if fb == 'f':
                    pml_position = "front"
                elif fb == 'b':
                    pml_position = "back"

                update_cpml(
                        in_out_field,
                        self.space.number_cells,
                        self.space.efield, # update_field
                        self.space.hfield, # base_field
                        self.space.ceb,
                        self.space.ds,
                        self.number_pml_cells,
                        self.pml_coefficient_e,
                        self.psi_e,
                        pml_direction,
                        pml_position)


    def update_cpml_h(self, pml_apply_opt):
        if self.space.grid_opt == 'efaced':
            in_out_field = "in_field"
        elif self.space.grid_opt == 'hfaced':
            in_out_field = "out_field"

        for pml_direction, position in enumerate(pml_apply_opt):    
            for fb in position:
                if fb == 'f':
                    pml_position = 'front'
                elif fb == 'b':
                    pml_position = 'back'

                update_cpml(
                        in_out_field,
                        self.space.number_cells,
                        self.space.hfield, # update_field
                        self.space.efield, # base_field
                        self.space.chb,
                        self.space.ds*self.space.unit_factor,
                        self.number_pml_cells,
                        self.pml_coefficient_h,
                        self.psi_h,
                        pml_direction,
                        pml_position)



#==============================================================================
# test code
#==============================================================================
if __name__ == '__main__':        
    from time import *
    from scipy import sin, sqrt

    ds = 10e-9
    number_cells = [200, 200, 30]
    grid_opt = 'efaced'
    unit_opt = 'Enorm'

    # construct the matter object
    '''
    # dielectric
    from kufdtd.dim3.matter.dielectric import Dielectric
    space = Dielectric(ds, number_cells, grid_opt, unit_opt)
    '''
    # drude
    from kufdtd.dim3.matter.drude import Drude
    space = Drude(ds, number_cells, grid_opt, unit_opt)
    space.set_drude_parameters(
            epr_infinity = 9.0685, 
            plasma_freq = 2*pi*2155.6e12/sqrt(9.0685), 
            gamma = 2*pi*18.36e12)	# modified P.R.B parameters

    space.allocate_arrays()
    Nx, Ny, Nz = number_cells[0], number_cells[1], number_cells[2]
    Ex, Ey, Ez = space.efield[0], space.efield[1], space.efield[2]
    space.set_coefficients()

    # construct the PML object
    number_pml_cells = 10
    kapa_max = 7
    alpha = 0.05
    grade_order = 4
    cpml_parameters = (kapa_max, alpha, grade_order)
    pml_space = Cpml(space, number_pml_cells, cpml_parameters)
    pml_space.allocate_pml_arrays()
    pml_apply_opt = ('fb', 'fb', 'fb') # front and back

    # for graphics using matplotlib
    from pylab import *
    ion()
    figure(figsize=(10,5))

    # for sin source
    wavelength = 300e-9
    wfreq = light_velocity*2*pi/wavelength#

    # for data capture
    cap_t = 10 # capture_time
    cap_pt = Nx/2 # capture_point

    #==========================================================================
    # main time loop
    #==========================================================================
    t0 = time()
    for tstep in xrange(100000):
        space.update_e()
        pml_space.update_cpml_e(pml_apply_opt)

        pulse = sin(wfreq*space.dt*tstep)
        space.efield[2][Nx/3, Ny/3*2, :] += pulse
        #space.efield[2][Nx/2, Ny/2, :] += pulse#

        space.update_h()
        pml_space.update_cpml_h(pml_apply_opt)

        if tstep/cap_t*cap_t == tstep:
            t1 = time()
            elapse_time = localtime(t1-t0-60*60*9)
            str_time = strftime('[%j]%H:%M:%S', elapse_time)
            print '%s    tstep = %d' % (str_time, tstep)

            intensity = (Ex[:,:,14]**2 + Ey[:,:,14] + Ez[:,:,14]**2)
            #clf()
            imshow(rot90(intensity),
                    cmap=cm.hot,
                    vmin=0, vmax=0.001,
                    interpolation='bilinear')
            #colorbar()
            draw()
