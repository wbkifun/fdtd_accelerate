#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : set_mp_area_arrays.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 4. 16

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the functions for conformal area arrays.
    set_mp_area_arrays
    set_overlap_boundary_grid_area
    make_cell_polygon_points
    overlap_area
    net_area

===============================================================================
"""

from structure_base import *

def set_mp_area_arrays(structures, MP_area_arrays, number_cells, \
        mpicoord=(0,0,0)): # for mpi
    '''
    Set grid-area value
    '''
    for axis in [x_axis, y_axis, z_axis]:
        axis_list = [x_axis, y_axis, z_axis]
        axis_list.pop(axis)
        axis_list.insert(2, axis)
        i_axis, j_axis, k_axis = axis_list

        for k in xrange( number_cells[k_axis] ):
            for i in xrange( number_cells[i_axis] ):
                for j in xrange( number_cells[j_axis] ):
                    polygon_points_list = []

                    for structure in structures:
                        ISPTs_arrays = structure.intersection_points_arrays 
                        MP = list(structure.matter)[1:]

                        i_shift = mpicoord[i_axis]*number_cells[i_axis]
                        j_shift = mpicoord[j_axis]*number_cells[j_axis]
                        k_shift = mpicoord[k_axis]*number_cells[k_axis]

                        polygon_points = make_cell_polygon_points( \
                                ISPTs_arrays, axis_list, \
                                i+i_shift, j+j_shift, k+k_shift)

                        polygon_points_list.append( polygon_points )

                    # find net_area each structure area
                    net_a = []
                    for l in xrange( len(polygon_points_list) ):
                        net_a.append( net_area(polygon_points_list[l:]) )

                    # apply the subcell MP_area_array value
                    slice_index = [i,j]
                    slice_index.insert(axis, k)
                    for m, MP_area_array in enumerate(MP_area_arrays):
                        sum_a_pec = 0
                        sum_mul_a_epr = 0
                        for l, area in enumerate(net_a):
                            MP = list(structures[l].matter)[1:] 

                            if MP[0] == sc.inf:
                                sum_a_pec += area
                            else:
                                sum_mul_a_epr += area*MP[m]

                        if sum_d_pec >= 1-0.0001:
                            MP_area_array[axis][tuple(slice_index)] = sc.inf
                            break
                        else:
                            MP_area_array[axis][tuple(slice_index)] = \
                                    sum_mul_a_epr/((1-sum_a_pec)**2)


def make_cell_polygon_points(ISPTs_arrays, axis_list, i, j, k):
    i_axis, j_axis, axis = axis_list

    if axis == x_axis:
        c1, c2, c3, c4 = k, j, k, j+1
        c5, c6, c7, c8 = k, i, k, i+1
    elif axis == y_axis:
        c1, c2, c3, c4 = k, j, k, j+1
        c5, c6, c7, c8 = i, k, i+1, k
    elif axis == z_axis:
        c1, c2, c3, c4 = j, k, j+1, k
        c5, c6, c7, c8 = i, k, i+1, k

    x1 = ISPTs_arrays[i_axis][0][c1, c2]
    x2 = ISPTs_arrays[i_axis][1][c1, c2]
    x3 = ISPTs_arrays[i_axis][0][c3, c4]
    x4 = ISPTs_arrays[i_axis][1][c3, c4]
    y1 = ISPTs_arrays[j_axis][0][c5, c6]
    y2 = ISPTs_arrays[j_axis][1][c5, c6]
    y3 = ISPTs_arrays[j_axis][0][c7, c8]
    y4 = ISPTs_arrays[j_axis][1][c7, c8]

    xy_list = [x1, x2, x3, x4, y1, y2, y3, y4]

    #------------------------------------------------------
    # points which construct a polygon in a grid cell
    #------------------------------------------------------
    polygon_points = []
    for n, x in enumerate( xy_list[:4] ):
        if int(x) == i:
            polygon_points.append( (x%1, n/2) )
        elif ( n%2 == 0 and int(x) < i ) or \
                ( n%2 == 1 and int(x) > i ):
            polygon_points.append( (n%2, n/2) )

    for n, y in enumerate( xy_list[4:] ):
        if int(y) == j:
            polygon_points.append( (n/2, y%1) )
        elif ( n%2 == 0 and int(y) < j ) or \
                ( n%2 == 1 and int(y) > j ):
            polygon_points.append( (n/2, n%2) )

    # for exception
    if len( polygon_points ) == 2:
        polygon_points.append( (0.5, 0.5) )

    # sort by rotation
    polygon_points = sort_points( polygon_points )

    return polygon_points


def overlap_area(polygon_points1, polygon_points2):
    spts1, spts2 = polygon_points1, polygon_points2

    overlap_polygon_points = []
    ext_spts1 = spts1+[spts1[0]]
    ext_spts2 = spts2+[spts2[0]]
    for i, p1 in enumerate( ext_spts1[:-1] ):
        p2 = ext_spts1[i+1]

        a1 = p2[1] - p1[1]
        b1 = -(p2[0] - p1[0])
        c1 = -a1*p1[0] - b1*p1[1]
        for j, p3 in enumerate( ext_spts2[:-1] ):
            p4 = ext_spts2[j+1]
            #print p1, p2, p3, p4

            a2 = p4[1] - p3[1]
            b2 = -(p4[0] - p3[0])
            c2 = -a2*p3[0] - b2*p3[1]

            D = a1*b2 - a2*b1
            if D != 0:
                ispt = ( ((-b2*c1 + b1*c2)/D), \
                         ((a2*c1 - a1*c2)/D) )

                #print '\tispt= ', ispt

                '''
                if point_in_boundary(spts1, ispt):
                    print 'point_in_boundary(spts1, ispt) is True'
                if point_in_boundary(spts2, ispt):
                    print 'point_in_boundary(spts2, ispt) is True'
                '''
                if point_in_boundary(spts1, ispt) and \
                        point_in_boundary(spts2, ispt) and \
                        ispt not in overlap_polygon_points:
                    overlap_polygon_points.append(ispt)

    #print 'overlap_polygon_points= ', overlap_polygon_points
    if len(overlap_polygon_points) == 0:
        oa = 0
        spts = [(0,0)]
    else:
        overlap_polygon_points = sort_points( overlap_polygon_points )
        oa = area_points(overlap_polygon_points)
        spts = overlap_polygon_points

    return oa, spts


def test_overlap_area():
    polygon_points1 = [[(0,0),(0.5,0),(0.5,1),(0,1)],\
            [(0,0),(5./12,0),(0.75,1),(0,1)],\
            [(0,0),(0.5,0),(3./4,1),(0,1)],\
            [(0,0),(1./2,0),(1./2,1),(0,1)]]
    polygon_points2 = [[(0,0),(1,0),(1,0.5),(0,0.5)],\
            [(0,0),(1,0),(1,0.5)],\
            [(0,1./4),(1,1./4),(1,3./4),(0,3./4)],\
            [(1./2,0),(1,0),(1,1),(3./4,1)]]
    print '\toverlap_area\toverlap_points' 
    for i, spts1 in enumerate(polygon_points1):
        spts2 = polygon_points2[i]
        oa, oa_spts = overlap_area(spts1, spts2)
        print '%d)\t%g\t\t%s' %(i, oa, oa_spts)


def net_area(polygon_points_list):
    spts_list = polygon_points_list

    N = len(spts_list)
    spts1 = spts_list.pop(0)
    oa1 = area_points(spts1)

    if N == 1:
        net_a = oa1

    elif N == 2:
        oa2, oa_spts2 = overlap_area(spts1, spts_list[0])

        net_a = oa1 - oa2

    elif N == 3:
        oa2, oa_spts2 = [], []
        for i, spts in enumerate(spts_list):
            oa, oa_spts = overlap_area(spts1, spts_list[i])
            oa2.append(oa)
            oa_spts2.append(oa_spts)

        oa3, oa_spts3 = overlap_area(oa_spts2[0], oa_spts2[1])

        net_a = oa1 - sum(oa2) + oa3

    elif N == 4:
        oa2, oa_spts2 = [], []
        for i, spts in enumerate(spts_list):
            oa, oa_spts = overlap_area(spts1, spts_list[i])
            oa2.append(oa)
            oa_spts2.append(oa_spts)

        oa3, oa_spts3 = [], []
        for i in xrange( len(oa2)-1 ):
            for j in xrange( i+1, len(oa2) ):
                oa, oa_spts = overlap_area(oa_spts2[i], oa_spts2[j])
                oa3.append(oa)
                oa_spts3.append(oa_spts)

        oa4, oa_spts4 = overlap_area(oa_spts3[0], oa_spts3[1])

        net_a = oa1 - sum(oa2) + sum(oa3) -oa4

    return net_a


def test_net_area():
    polygon_points_list_list = [\
            [ [(0,0),(0.5,0),(0.5,1),(0,1)],\
                [(0,0),(1,0),(1,0.5),(0,0.5)] ], \
            [ [(0,0),(0.5,0),(0.5,1),(0,1)],\
                [(0,0),(1,0),(1,0.25),(0,0.25)] ] \
            ]
    print '\tnet_area' 
    for i, sptsl in enumerate(polygon_points_list_list):
        print '%d)\t%g' %(i, net_area(sptsl))



if __name__ == '__main__':
    #test_overlap_area()
    test_net_area()
