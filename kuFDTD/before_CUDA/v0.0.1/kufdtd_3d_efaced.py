#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : kufdtd_3d_efaced.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Written date : 2008. 01. 03. Thr

 Modifier : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Modified date : 2008. 01. 10. Thr

 Copyright : This has used lots of python modules which is opend to public. So,
 it is also in pulic.

============================== < File Description > ===========================

이 파일은 KUFDTD(Korea University Finite Difference Time Domain method)의
 3D 계산을 위해 필요한 클래스들을 모은 파일이다. 이 파일 안에는 3D dielectric
 calculation, 3D metal calculation, 3D cpml calculation class들이 들어있다.

===============================================================================
"""

from kufdtd_base import *
import kufdtd_3d_efaced_core as core_3de

def matter_average(matter, matter_base):
    """
    """
    a = slice(None, None, None)  # all cells in the direction = [:]
    p = slice(1, None, None)  # from the index 1 to end = [1:]
    n = slice(None, -1, None)  # from the index 0 to -1 = [:-1]
    if matter[x_axis] == ArrayType:
        matter[x_axis][p, a, a] =\
                0.5 * (matter_base[p, a, a] + matter_base[n, a, a])
    if matter[y_axis] == ArrayType:
        matter[y_axis][a, p, a] =\
                0.5 * (matter_base[a, p, a] + matter_base[a, n, a])
    if matter[z_axis] == ArrayType:
        matter[z_axis][a, a, p] =\
                0.5 * (matter_base[a, a, p] + matter_base[a, a, n])
    return matter
_

class DielectricEfaced3dBase(FdtdSpace):
    """
    """
    def __init__(self, number_cells, ds, total_length, data_type='f'):
        """
        """
        FdtdSpace.__init__(self, ds, total_length, dim_3d)
        self.dtyp = data_type
        self.number_cells = sc.array(number_cells)
        # self.efield is (ex, ey, ez) and self.hfield is (hx, hy, hz).
        e_number_cells = number_cells + sc.array([2, 2, 2])
        h_number_cells = number_cells + sc.array([1, 1, 1])
        self.efield = define_space_array(e_number_cells, 0, self.dtyp, 'xyz')
        self.hfield = define_space_array(h_number_cells, 0, self.dtyp, 'xyz')
        self.epsr = define_space_array(e_number_cells, 1, self.dtyp, 'xyz')
        self.epsr_base = sc.ones(e_number_cells, self.dtyp)

    def set_coefficient(self):
        """
        """
        self.epsr = matter_average(self.epsr, self.epsr_base)
        del self.epsr_base
        temp_ceb = calc_with_list(2, '*', self.epsr)
        del self.epsr
        self.ceb = calc_with_list(1., '/', temp_ceb)
        del temp_ceb
        self.chb = 0.5

    def updateE(self):
        """
        """
        core_3de.dielectric_updateE(self.number_cells, self.efield,\
                self.hfield, self.ceb)

    def updateH(self):
        """
        """
        core_3de.dielectric_updateH(self.number_cells, self.efield,\
                self.hfield, self.chb)


class DrudeMetalEfaced3dBase(FdtdSapce):
    """
    """
    def __init__(self, number_cells, ds, total_length, data_type='f'):
        """
        """
        FdtdSpace.__init__(self, ds, total_length, dim_3d)
        self.dtyp = data_type
        self.number_cells = sc.array(number_cells)
        e_number_cells = number_cells + sc.array([2, 2, 2])
        h_number_cells = number_cells + sc.array([1, 1, 1])
        self.efield = define_space_array(e_number_cells, 0, self.dtyp, 'xyz')
        self.hfield = define_space_array(h_number_cells, 0, self.dtyp, 'xyz')
        self.fefield = define_space_array(e_number_cells, 0, self.dtyp, 'xyz')
        self.epsr_inf = define_space_array(e_number_cells, 1, 'd', 'xyz')
        self.pfreq = define_space_array(e_number_cells, 0, 'd', 'xyz')
        self.gamma = define_space_array(e_number_cells, 0, 'd', 'xyz')
        self.epsr_inf_base = sc.ones(e_number_cells, 'd')
        self.pfreq_base = sc.zeros(e_number_cells, 'd')
        self.gamma_base = sc.zeros(e_number_cells, 'd')

    def set_coefficient(self):
        """
        """
        self.epsr_inf = matter_average_3d(self.epsr_inf, self.epsr_inf_base)
        del self.epsr_inf_base
        self.pfreq = matter_average_3d(self.pfreq, self.pfreq_base)
        del self.pfreq_base
        self.gamma = matter_average_3d(self.gamma, self.gamma_base)
        del self.gamma_base
        square_pfreq = calc_with_list(self.pfreq, '**' ,2)
        del self.pfreq
        temp_cea = calc_with_list(square_pfreq, '*', self.dt**2)
        temp_cea = calc_with_list(2., '+', temp_cea)
        self.cea = calc_with_list(1., '/' ,temp_cea)
        del temp_cea
        cef_temp1 = calc_with_list(self.cea, '*', square_pfreq)
        cef_temp2 = []
        for axis in xrange(dim_3d):
            cef_temp2.append(self.dt*(sc.exp(-self.gamma[axis] * self.dt) + 1.))
        self.cef = cef_temp1 * cef_temp2
        del cef_temp1
        del cef_temp2
        del square_pfreq
        self.ceb = self.cea / self.epsr_inf
        del self.epsr_inf
        self.cea = self.cea.astype(self.dtyp)  # cast from double to float
        self.cef = self.cef.astype(self.dtyp)
        self.ceb = self.ceb.astype(self.dtyp)
        self.chb = 0.5

    def updateE(self):
        """
        """
        core_3de.drude_metal_updateE(self.number_cells, self.efield,\
                self.hfield, self.fefield, self.cea, self.cef, self.ceb,\
                self.gamma, self.dt)

    def updateH(self):
        """
        """
        core_3de.drude_metal_updateH(self.number_cells, self.efield,\
                self.hfield, self.chb)


class CpmlEfaced:
    """
    이 클래스는 가상 클래스로 하위 구조에서 사용할 변수들과 method만 정의하여
    놓았다. 즉 이 클래스는 인스턴스를 생성하지 않는다.
    """
    def __init__(self, ds, number_pml_cells=10, kapa_max=7., alpha=0.05,\
            grade_order=4, data_type='f'):
        """
        """
        self.dtyp = data_type
        dt = ds / (2 * light_velocity)
        self.npml = number_pml_cells  # depth of PML cell
        g_order = grade_order
        sigma_max = (g_order + 1) / (150 * pi * ds)

        sigma_E = self.make_cpml_array(npml, 'half_depth')
        sigma_H = self.make_cpml_array(npml, 'full_depth')
        sigma_E = pow(sigma_E / npml, g_order) * sigma_max
        sigma_H = pow(sigma_H / npml, g_order) * sigma_max 
        self.kapa_E = self.make_cpml_array(npml, 'half_depth')
        self.kapa_H = self.make_cpml_array(npml, 'full_depth')
        self.kapa_E = 1. + (kapa_max - 1) * pow(self.kapa_E / npml, g_order)
        self.kapa_H = 1. + (kapa_max - 1) * pow(self.kapa_H / npml, g_order)

        self.rcm_bE = sc.exp(-(sigma_E / self.kapa_E + alpha) * dt / ep0)
        self.mrcm_aE = (sigma_E / (sigma_E * self.kapa_E + alpha *\
                self.kapa_E**2) * (self.rcm_bE - 1)) / dx
        self.rcm_bH = sc.exp(-(sigma_H / self.kapa_H + alpha) * dt / ep0)
        self.mrcm_aH = (sigma_H / (sigma_H * self.kapa_H + alpha *\
                self.kapa_H**2) * (self.rcm_bH - 1)) * 0.5 * mu0 / dt
        self.cpml_e_coefficient = [self.rcm_bE, self.mrcm_aE, self.kapa_E]
        self.cpml_h_coefficient = [self.rcm_bH, self.mrcm_aH, self.kapa_H]

    def make_cpml_array(s, number_pml_cells, interval):
        """make a array as symmertry"""
        N = number_pml_cells
        if interval == 'half_depth':
            cpml_array = sc.arange(N - 0.5, -N - 0.5, -1, self.dtyp)
            cpml_array[-N:] = abs(cpml_array[-N:])
        elif interval == 'full_depth':
            cpml_array = arange(N, -N, -1, self.dtyp)
            cpml_array[-N:] = abs(cpml_array[-N:]) + 1
        return cpml_array

        DielectricEfaced3dBase.__init__(self, number_cells, ds, total_length,\
                data_type)


class CpmlEfaced3d(CpmlEfaced):
    """
    이 클래스는 가상 클래스로 하위 구조에서 사용할 변수들과 method만 정의하여
    놓았다. 즉 이 클래스는 인스턴스를 생성하지 않는다.
    """
    def __init__(self, ds, number_pml_cells=10, kapa_max=7., alpha=0.05,\
            grade_order=4, data_type='f'):
        """
        """
        CpmlEfaced. __init__(self, ds, number_pml_cells, kapa_max, alpha,\
                grade_order, data_type):

    def make_psi_array(self):
        """
        """
        nx, ny, nz = self.number_cells
        np = self.npml
        self.psi_eyx = sc.zeros((2 * np, ny + 2, nz + 2), self.dtyp)
        self.psi_ezx = sc.zeros((2 * np, ny + 2, nz + 2), self.dtyp)
        self.psi_exy = sc.zeros((nx + 2, 2 * np, nz + 2), self.dtyp)
        self.psi_ezy = sc.zeros((nx + 2, 2 * np, nz + 2), self.dtyp)
        self.psi_exz = sc.zeros((nx + 2, ny + 2, 2 * np), self.dtyp)
        self.psi_eyz = sc.zeros((nx + 2, ny + 2, 2 * np), self.dtyp)
        self.psi_e = [self.psi_eyx, self.psi_ezx, self.psi_ezy,\
               self.psi_exy, self.psi_exz, self.psi_eyz]
        self.psi_hyx = sc.zeros((2 * np, ny + 1, nz + 1), self.dtyp)
        self.psi_hzx = sc.zeros((2 * np, ny + 1, nz + 1), self.dtyp)
        self.psi_hxy = sc.zeros((nx + 1, 2 * np, nz + 1), self.dtyp)
        self.psi_hzy = sc.zeros((nx + 1, 2 * np, nz + 1), self.dtyp)
        self.psi_hxz = sc.zeros((nx + 1, ny + 1, 2 * np), self.dtyp)
        self.psi_hyz = sc.zeros((nx + 1, ny + 1, 2 * np), self.dtyp)
        self.psi_h = [self.psi_hyx, self.psi_hzx, self.psi_hzy,\
                self.psi_hxy, self.psi_hxz, self.psi_hyz]

    def cpml_xE(self, applying):
        """
        """
        if 'f' in applying:
            # pml_p is the information of position of PML.
            pml_p = 0
            core_3de.cpml_xE(pml_p, self.npml, self.ds, self.number_cells,\
                    self.efield, self.hfield, self.psi_e, self.ceb,\
                    self.cpml_e_coefficient)
        if 'b' in applying:
            # pml_p is the information of position of PML.
            pml_p = 1
            core_3de.cpml_xE(pml_p, self.npml, self.ds, self.number_cells,\
                    self.efield, self.hfield, self.psi_e, self.ceb,\
                    self.cpml_e_coefficient)

    def cpml_yE(self, applying):
        """
        """
        if 'f' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 0
            core_3de.cpml_yE(pml_p, self.npml, self.ds, self.number_cells,\
                    self.efield, self.hfield, self.psi_e, self.ceb,\
                    self.cpml_e_coefficient)
        if 'b' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 1
            core_3de.cpml_yE(pml_p, self.npml, self.ds, self.number_cells,\
                    self.efield, self.hfield, self.psi_e, self.ceb,\
                    self.cpml_e_coefficient)

    def cpml_zE(self, applying):
        """
        """
        if 'f' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 0
            core_3de.cpml_zE(pml_p, self.npml, self.ds, self.number_cells,\
                    self.efield, self.hfield, self.psi_e, self.ceb,\
                    self.cpml_e_coefficient)
        if 'b' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 1
            core_3de.cpml_zE(pml_p, self.npml, self.ds, self.number_cells,\
                    self.efield, self.hfield, self.psi_e, self.ceb,\
                    self.cpml_e_coefficient)

    def cpml_xH(self, applying):
        """
        """
        if 'f' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 0
            core_3de.cpml_xH(pml_p, self.npml, self.dt, mu0,\
                    self.number_cells, self.efield, self.hfield, self.psi_h,\
                    self.chb, self.cpml_h_coefficient)
        if 'b' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 1
            core_3de.cpml_xH(pml_p, self.npml, self.dt, mu0,\
                    self.number_cells, self.efield, self.hfield, self.psi_h,\
                    self.chb, self.cpml_h_coefficient)

    def cpml_yH(self, applying):
        """
        """
        if 'f' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 0
            core_3de.cpml_yH(pml_p, self.npml, self.dt, mu0,\
                    self.number_cells, self.efield, self.hfield, self.psi_h,\
                    self.chb, self.cpml_h_coefficient)
        if 'b' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 1
            core_3de.cpml_yH(pml_p, self.npml, self.dt, mu0,\
                    self.number_cells, self.efield, self.hfield, self.psi_h,\
                    self.chb, self.cpml_h_coefficient)

    def cpml_zH(self, applying):
        """
        """
        if 'f' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 0
            core_3de.cpml_zH(pml_p, self.npml, self.dt, mu0,\
                    self.number_cells, self.efield, self.hfield, self.psi_h,\
                    self.chb, self.cpml_h_coefficient)
        if 'b' in applying:
            # pml_p is the information of directon and position of PML.
            pml_p = 1
            core_3de.cpml_zH(pml_p, self.npml, self.dt, mu0,\
                    self.number_cells, self.efield, self.hfield, self.psi_h,\
                    self.chb, self.cpml_h_coefficient)


class DielectricEfaced3dCpml(DielectricEfaced3dBase, CpmlEfaced3d):
    """
    """
    def __init__(self, number_cells, ds, total_length, number_pml_cells=10,\
            kapa_max=7., alpha=0.05, grade_order=4, data_type='f'):
        """
        """
        DielectricEfaced3dBase.__init__(self, number_cells, ds, total_length,\
                data_type)
        CpmlEfaced3d.__init__(self, ds, number_pml_cells, kapa_max, alpha,\
                grade_order, data_type)


class DrudeMetalEfaced3dCpml(DrudeMetalEfaced3dBase, CpmlEfaced3d):
    """
    """
    def __init__(self, number_cells, ds, total_length, number_pml_cells=10,\
            kapa_max=7., alpha=0.05, grade_order=4, data_type='f'):
        """
        """
        MetalEfaced3dBase.__init__(self, number_cells, ds, total_length,\
                data_type)
        CpmlEfaced3d.__init__(self, ds, number_pml_cells, kapa_max, alpha,\
                grade_order, data_type)
