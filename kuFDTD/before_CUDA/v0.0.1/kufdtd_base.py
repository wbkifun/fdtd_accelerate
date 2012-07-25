#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : kufdtd_base.py

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Written date : 2008. 01. 02. Wed

 Modifier : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Modified date : 2008. 01. 10. Thr

 Copyright : This has used lots of python modules which is opend to public. So,
 it is also in pulic.

============================== < File Description > ===========================

이 파일은 KUFDTD(Korea University Finite Difference Time Domain method)의
기본을 이루는 물리적인 변수, 수학적인 정의, 공간적인 정의, 전산모사에서
필요로 하는 pre-defined 변수, 그리고 KUFDTD에서 유용하게 쓰일 파이썬 함수들을
선언한다.

===============================================================================
"""

import scipy as sc


# 물리 상수들
light_velocity = 2.99792458e8  # m s-, 0.0 (빛의 속도)
ep0 = 8.85418781762038920e-12  # F m-1, 0.0 (진공에서의 유전률)
mu0 = 1.25663706143591730e-6  # N A-2, 0.0 (진공에서의 투자율)
imp0 = sc.sqrt(ep0 / mu0)  # 진공에서의 임피던스
h = 4.13566727e-15  # eV s, 3.9e-8 (플랑크 상수)
hBar = 6.58211889e-16  # eV s, 3.9e-8 (플랑크 상수를 2pi로 나눈 값)
pi = sc.pi  # 원주율 파이
# 직각좌표계에서 각 방향을 0, 1, 2 으로 정의하였다.
x_axis = 0
y_axis = 1
z_axis = 2
# 구형좌표계에서 각 방향을 0, 1, 2 으로 정의하였다.
r_axis = 0
th_axis = 1
ph_axis = 2
#전산모사와 관련하여 편의를 위한 정의
ArrayType = type(sc.array([0, 0]))
ListType = type([0,0])
dim_3d = 3
dim_2d = 2
dim_1d = 1
server = 0  # mpi에서 server의 가독성을 위해 추가
# 파이썬과 관련하여 편의를 위한 정의
def calc_with_list(value1, operator ,value2):  ## 테스트 코드 필요함.
    """
    """
    result_list = []
    value1_ty = type(value1)
    value2_ty = type(value2)
    if operator not in ['+','-','*','/','**']:
        raise Exception, 'There is no mached operator.\
                Please use +, -, *, /, **.'
    if value1_ty is not ListType and value2_ty is not ListType:
        raise Exception, 'Both variables are not List. Just use %s Operator'\
                % operator
    if value1_ty is not ListType:
        operation_str = 'value1%smem'  % operator
        for mem in value2:
            pass
            result_list.append(eval(operation_str))
    elif value2_ty is not ListType:
        operation_str = 'mem%svalue2' % operator
        for mem in value1:
            pass
            result_list.append(eval(operation_str))
    else:
        if len(value1) == len(value2):
            operation_str = 'mem%svalue2[i]' % operator
            for i, mem in enumerate(value1):
                result_list.append(eval(operation_str))
        else:
            raise Exception, 'The sizes of both lists are not equal.'
    return result_list

class FdtdSpace:
    """
    """
    def __init__(self, ds, total_length, dimension=dim_3d):
        """
        """
        self.ds = ds  # cell size
        self.total_length = sc.array(total_length,'d')  # = (Lx, Ly, Lz)
        if dimension == dim_2d:
            if self.total_length[z_axis] != self.ds:
                raise Exception, 'The dimension of the simulation is 2D.\
                However, the length of z is not %s.' % self.ds
        elif dimension == dim_1d:
            if self.total_length [z_axis] != self.ds and\
                    self.total_length[y_axis] != self.ds:
                raise Exception, 'The dimension of the simulation is 1D.\
                However, the length of z and the length of y are not %s.'\
                % self.ds
        #self.origin_real = self.total_length * 0.5
        self.dimension = dimension
        self.total_number_cells = sc.array(self.real2discrete(ds, total_length))
        #self.origin_cell = sc.array(self.real2discrete(ds, self.origin_real))
        self.dt = ds / (2 * light_velocity)
    def __repr__(self):
        """
        """
        return """
        Cell Size(ds) = %.2e
        Infinitesimal Time(dt) = %.2e
        Dimension of the Simulation = %s
        Total length of the Space = %s
        The number of the total cells = %s
        The Origin of the cells = %s
        The Origin of the Space = %s""" % (self.ds, self.dt,
                self.dimension, self.total_length, self.total_number_cells,
                self.origin_real, self.origin_cell)

    def real2discrete(self, ds, *real_value):
        """
        이름 : real2discrete
        기능 : 이 함수는 프로젝트 파일에서 real 공간 값으로 선언된 것을
               전산모사에서 계산을 할 수 있도록 discrete 한 공간 값으로 변환해
               준다. 따라서 주어지는 인자는 한 cell의 최소 단위인 dx가
               주어지고, 변환하고자 하는 real 값들을 넣어준다.
        인자 : dx, real_vals
        반환 : discrete_val

        보충 : 들어오는 real 값 개수에 맞추어 결과 discrete 값의 개수도
               반환된다. 단지 한 개가 아닌 여러개 인자인 경우 tuple로 반환이
               된다.
        """
        number_members = len(real_value)
        list_discrete_value = []
        for mem in xrange(number_members):
            array_real_value = sc.array(real_value[mem])
            array_real_value = array_real_value / ds
            discrete_value = sc.round_(array_real_value)
            discrete_value = sc.array(discrete_value,'i')
            list_discrete_value.append(discrete_value)
        if len(list_discrete_value) == 1 :
            return list_discrete_value[0]
        else:
            return list_discrete_value

def define_space_array(number_cells, initial_value=0, data_type='f',
        space_direction='xyz'):  ## 테스트 코드 필요함
    """
    """
    space = [ones((1,), data_type), ones((1,), data_type),\
             ones((1,), data_type)]
    number_cells = tuple(number_cells)
    if inital_value != 0 or initial_value != 1:
            raise Exception, 'The initial value must be zero or one'
    if initial_value:
        if 'x' in space_direction:
            space[x_axis] = sc.ones(number_cells, data_type)
        if 'y' in space_direction:
            space[y_axis] = sc.ones(number_cells, data_type)
        if 'z' in space_direction:
            space[z_axis] = sc.ones(number_cells, data_type)
    else:
        if 'x' in space_direction:
            space[x_axis] = sc.zeros(number_cells, data_type)
        if 'y' in space_direction:
            space[y_axis] = sc.zeros(number_cells, data_type)
        if 'z' in space_direction:
            space[z_axis] = sc.zeros(number_cells, data_type)
    return space

def del_non_used_space_array(array_set, axis, data_type):  ## 테스트 코드 필요함
    """
    """
    if 'x' in axis:
        array_set[x_axis] = ones((1,), data_type)
    if 'y' in axis:
        array_set[y_axis] = ones((1,), data_type)
    if 'z' in axis:
        array_set[z_axis] = ones((1,), data_type)
    return array_set    



if __name__ == '__main__':
    """이 부분은 kufdtd_base를 테스트 하기 위한 부분이다."""
    print '==================================='
    point1 = (2, 45, 0)
    point2 = (1, 0, 90)
    point3 = (4, 45, 180)
    point4 = (1, 60, 270)
    vec1 = Vector(point1,polar=True, degree=True)
    print vec1
    vec2 = Vector(point2,polar=True, degree=True)
    print vec2
    vec3 = Vector(point3,polar=True, degree=True)
    print vec3
    vec4 = Vector(point4,polar=True, degree=True)
    print vec4
    print '==================================='
    point11 = (-2, -2, 2)
    point22 = (-2, 2, 2)
    point33 = (2, -2, 2)
    point44 = (2, 2, -2)
    vec1 = Vector(point11, point1)
    print vec1
    vec2 = Vector(point22, point2)
    print vec2
    vec3 = Vector(point33, point3)
    print vec3
    vec4 = Vector(point44, point4)
    print vec4
    print '==================================='
    point1 = (-1, -1, 1)
    point2 = (-1, 1, 1)
    point3 = (1, -1, 1)
    point4 = (1, 1, -1)
    vec1 = Vector(point1)
    print vec1
    vec2 = Vector(point2)
    print vec2
    vec3 = Vector(point3)
    print vec3
    vec4 = Vector(point4)
    print vec4
    print '==================================='
    point = (1, 1, 1)
    vec = Vector(point)
    print vec
    vec.move((1, 1, 1))
    print vec
    vec.rotate((30,20),degree=True)
    print vec
    vec.rotate((-30,-20),degree=True)
    print vec
    print '==================================='
    fs = FdtdSpace(5e-9, (100e-9, 200e-9, 150e-9))
    print fs