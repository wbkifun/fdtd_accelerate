#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : set_mp_line_arrays.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 4. 10

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the functions for conformal line arrays.
    set_mp_line_arrays
    set_overlap_boundary_grid_line
    overlap_distance
    net_distance

===============================================================================
"""

from structure_base import *

def set_mp_line_arrays(structures, MP_line_arrays, number_cells, \
        mpicoord=(0,0,0)): # for mpi
    '''
    Set grid-line value
    '''
    for axis in [x_axis, y_axis, z_axis]:
        axis_list = [x_axis, y_axis, z_axis]
        axis_list.pop(axis)
        i_axis, j_axis = axis_list

        for i in xrange( number_cells[i_axis] ):
            for j in xrange( number_cells[j_axis] ):
                x_list = []
                i_list = []

                for structure in structures:
                    ISPTs_arrays = structure.intersection_points_arrays
                    MP = list(structure.matter)[1:]

                    i_shift = mpicoord[i_axis]*number_cells[i_aixs]
                    j_shift = mpicoord[j_axis]*number_cells[j_aixs]
                    x1 = ISPTs_arrays[axis][0][i+i_shift, j+j_shift]
                    x2 = ISPTs_arrays[axis][1][i+i_shift, j+j_shift]

                    i1, i2 = int(x1), int(x2)

                    #----------------------------------------------------------
                    #Set main cell except for boundary cell
                    #----------------------------------------------------------
                    mpi_i1 = mpicoord[axis]*number_cells[axis]
                    mpi_i2 = (mpicoord[axis] + 1)*number_cells[axis] - 1

                    if i1 < mpi_x1:
                        slice_1 = None
                    elif i1 >= mpi_i1 and i1 < mpi_i2:
                        slice_1 = i1+1

                    if i2 > mpi_i2:
                        slice_2 = None
                    elif i2 > mpi_i1 and i2 <= mpi_i2:
                        slice_2 = i2

                    if i1 != i2:
                        slice_indices = [i,j]
                        slice_indices.insert(axis, slice(slice_1, slice_2))
                        for m, MP_line_array in enumerate(MP_line_arrays):
                            MP_line_array[axis][tuple(slice_indices)] = MP[m]

                    #for set boundary cell
                    x_list.append(x1)
                    x_list.append(x2)
                    i_list.append(i1)
                    i_list.append(i2)

                #--------------------------------------------------------------
                # set boundary cell 
                #--------------------------------------------------------------
                for k, ival in enumerate(i_list):
                    # Is there the ival between a high priority order?
                    included = False
                    for l in xrange( (k/2+1)*2, len(i_list)-1, 2):
                        l1, l2 = i_list[l], i_list[l+1]
                        if (l1-ival)*(l2-ival) < 0:
                            included = True
                            break

                    if included:
                        pass
                    # Is there the ival in my mpi-region?
                    elif (mpi_i1-ival)*(mpi-i2-ival) > 0:
                        pass
                    # Is there a same ival under my index?
                    elif ival in i_list[:k]:
                        pass
                    else:
                        # remapping x coord. into the boundary cell
                        structure_index = []
                        sx_list = []
                        for l in xrange( len(i_list)/2 ):
                            i1 = i_list[2*l]
                            i2 = i_list[2*l+1]
                            if i1 <= k and i2 >= k:
                                structure_index.append(l)
                                sx_list.append( x_list[2*l]%1 )
                                sx_list.append( x_list[2*l+1]%1 )

                                if i1 < k:
                                    sx_list[2*l] = 0
                                if i2 > k:
                                    sx_list[2*l+1] = 1

                        # find net_distance each structure line
                        net_d = []
                        for l in xrange( len(sx_list)/2 ):
                            net_d.append( net_distance(sx_list[2*l:]) )

                        # apply the subcell MP_line_array value
                        slice_index = [i,j]
                        slice_index.insert(axis, k)
                        for m, MP_line_array in enumerate(MP_line_arrays):
                            sum_d_pec = 0
                            sum_mul_d_epr = 0
                            for l, d in enumerate(net_d):
                                MP = list(structures[ structure_index[l] ].matter)[1:]
                                if MP[0] == sc.inf:
                                    sum_d_pec += d
                                else:
                                    sum_mul_d_epr += d*MP[m]

                            if sum_d_pec >= 1-0.0001:
                                MP_line_array[axis][tuple(slice_index)] = sc.inf
                                break
                            else:
                                MP_line_array[axis][tuple(slice_index)] = \
                                        sum_mul_d_epr/((1-sum_d_pec)**2)


def overlap_distance(d1, d2, cx1, cx2):
    D = abs(cx2 - cx1)
    #print '\tD = %g' % D
    #print '\t0.5*(d2+d1) = %g' %(0.5*(d2+d1))
    #print '\t0.5*(d2-d1) = %g' %(0.5*(d2-d1))

    if D >= 0.5*(d2+d1):
        #print '\tenter (1)'
        od, ocx = 0, 0

    elif D <= 0.5*abs(d2-d1):
        #print '\tenter (2)'
        if d1 < d2:
            od, ocx = d1, cx1
        else:
            od, ocx = d2, cx2

    else:
        #print '\tenter (3)'
        od = 0.5*(d2+d1) - D
        if cx1 < cx2:
            ocx = cx2 - 0.5*(d2-od)
        else:
            ocx = cx1 - 0.5*(d1-od)

    return od, ocx


def net_distance(d_list, cx_list):
    N = len(d_list)
    d1, cx1 = d_list.pop(0), cx_list.pop(0)

    if N == 1:
        net_d = d1

    elif N == 2:
        od2, ocx2 = overlap_distance(d1, d_list[0], cx1, cx_list[0])

        net_d = d1 - od2
    
    elif N == 3:
        od2, ocx2 = [], []
        for i, d in enumerate(d_list):
            od, ocx = overlap_distance(d1, d, cx1, cx_list[i])
            od2.append(od)
            ocx2.append(ocx)

        od3, ocx3 = overlap_distance(od2[0], od2[1], ocx2[0], ocx2[1])

        net_d = d1 - sum(od2) + od3

    elif N == 4:
        od2, ocx2 = [], []
        for i, d in enumerate(d_list):
            od, ocx = overlap_distance(d1, d, cx1, cx_list[i])
            od2.append(od)
            ocx2.append(ocx)

        od3, ocx3 = [], []
        for i in xrange( len(od2)-1 ):
            for j in xrange( i+1, len(od2) ):
                od, ocx = overlap_distance(od2[i], od2[j], ocx2[i], ocx2[j])
                od3.append(od)
                ocx3.append(ocx)

        od4, ocx4 = overlap_distance(od3[0], od3[1], ocx3[0], ocx3[1])

        net_d = d1 - sum(od2) + sum(od3) - od4

    return net_d


def test_overlap_distance():
    '''
    test for overlap_distance()
    '''
    d1, d2 = 0.4, 0.3
    l_cx1 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.6, 0.7]
    l_cx2 = [0.75, 0.65, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25]

    l_exact_od = [0, 0, 0.1, 0.3, 0.3, 0.1, 0, 0]
    l_exact_ocx = [0, 0, 0.45, 0.35, 0.25, 0.35, 0, 0]

    print '-'*70
    print '\td1\td2\tcx1\tcx2  | numeric\t\t| exact'
    print '-'*70
    for i, cx1 in enumerate(l_cx1):
        numeric_od, numeric_ocx = overlap_distance(d1, d2, cx1, l_cx2[i])

        print '%d)\t%g\t%g\t%g\t%g | %g\t%g\t| %g\t%g' %\
                (i+1, d1, d2, cx1, l_cx2[i], numeric_od, numeric_ocx, l_exact_od[i], l_exact_ocx[i])


def test_net_distance():
    '''
    test for overlap_distance()
    '''
    l_exact_net_d = [0.6, 0.6, 0.3, 0.1, 0, 0, 0.2, 0.2, 0.15, 0.5, 0.6]
    l_d_list = [[0.6], [0.6, 0.3], [0.6, 0.5], [0.6, 0.5], [0.5, 0.5], \
            [0.4, 0.5], [0.4, 0.4, 0.3], [0.7, 0.3, 0.3], \
            [0.7, 0.3, 0.3, 0.15], [1, 0.5], [1, 0.4]]
    l_cx_list = [[0.3], [0.3, 0.85], [0.3, 0.55], [0.3, 0.35], [0.35, 0.35], \
            [0.4, 0.35], [0.3, 0.5, 0.75], [0.45, 0.35, 0.55], \
            [0.45, 0.35, 0.55, 0.825], [0.5, 0.75], [0.5, 0.5]]

    print '-'*70
    print '\tnumeric\t|\texact'
    print '-'*70
    for i in xrange(11):
        numeric_net_d = net_distance(l_d_list[i], l_cx_list[i])

        print '%d)\t%g\t|\t%g' %(i+1, numeric_net_d, l_exact_net_d[i])

    #numeric_net_d = net_distance([0.4, 0.5], [0.4, 0.35])
    #print '%d)\t%g\t|\t%g' %(6, numeric_net_d, 0)
    #numeric_net_d = net_distance([0.4, 0.4, 0.3], [0.3, 0.5, 0.75])
    #print '%d)\t%g\t|\t%g' %(7, numeric_net_d, 0)
    #numeric_net_d = net_distance([0.4, 0.4, 0.3, 0.15], [0.3, 0.5, 0.75, 0.825])
    #print '%d)\t%g\t|\t%g' %(9, numeric_net_d, 0)



if __name__ == '__main__':
    test_overlap_distance()
    test_net_distance()
