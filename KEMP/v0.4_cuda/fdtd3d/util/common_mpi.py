import numpy as np

import common


def check_mpi_shape(size, mpi_shape):
    common.check_type('mpi_shape', mpi_shape, (list, tuple), int)

    assert size == reduce(lambda x, y: x*y, mpi_shape), \
            'MPI size %d is not matched with the mpi_shape %s.' % (size, repr(mpi_shape))



def my_coord(rank, mpi_shape):
    common.check_type('mpi_shape', mpi_shape, (list, tuple), int)

    mx, my, mz = mpi_shape
    return [rank%mx, rank/mx%my, rank/(mx*my)]



def mpi_target_dict(rank, mpi_shape, pbc_axes):
    common.check_type('mpi_shape', mpi_shape, (list, tuple), int)
    common.check_type('pbc_axes', pbc_axes, str)

    mx, my, mz = mpi_shape
    mpi_target_dict = {
            'x-': None, 'x+': None, \
            'y-': None, 'y+': None, \
            'z-': None, 'z+': None}

    mycoord = my_coord(rank, mpi_shape)
    replace = lambda i, val: mycoord[:i] + [val] + mycoord[i+1:]
    coord_to_rank = lambda (i, j, k): i + j*mx + k*mx*my

    for i, axis in zip([0, 1, 2], ['x', 'y', 'z']):
        val = mycoord[i]
        ms = mpi_shape[i]

        if val > 0:
            mpi_target_dict['%s-' % axis] = coord_to_rank(replace(i, val-1))
        elif val == 0 and axis in pbc_axes and ms != 1:
            mpi_target_dict['%s-' % axis] = coord_to_rank(replace(i, ms-1))

        if val < ms-1:
            mpi_target_dict['%s+' % axis] = coord_to_rank(replace(i, val+1))
        elif val == ms-1 and axis in pbc_axes and ms != 1:
            mpi_target_dict['%s+' % axis] = coord_to_rank(replace(i, 0))

    return mpi_target_dict



def accum_sub_ns_dict(mpi_shape, ndev, dnx_list, ny_list, nz_list):
    common.check_type('mpi_shape', mpi_shape, (tuple, list), int)
    common.check_type('ndev', ndev, int)
    common.check_type('dnx_list', dnx_list, (tuple, list), int)
    common.check_type('ny_list', ny_list, (tuple, list), int)
    common.check_type('nz_list', nz_list, (tuple, list), int)

    mx, my, mz = mpi_shape

    snx_list = []
    strip_dnx_list = []
    for mi in xrange(mx):
        sub_dnx_list = dnx_list[mi*ndev:(mi+1)*ndev]
        snx_list.append( sum(sub_dnx_list) - ndev + 1 )
        strip_dnx_list.extend( [nx-1 for nx in sub_dnx_list] )
        strip_dnx_list[-1] += 1

    accum_sub_ns_dict = { \
            'x': np.add.accumulate([0] + snx_list), \
            'y': np.add.accumulate([0] + ny_list), \
            'z': np.add.accumulate([0] + nz_list), \
            'dx': np.add.accumulate([0] + strip_dnx_list) }

    return accum_sub_ns_dict



def divide_info_dict(size, mpi_shape, pt0, pt1, asn_dict):
    info_dict = {}
    rank_list = []

    is_first = True
    for rank in xrange(size):
        axes = ['x', 'y', 'z']
        coord = my_coord(rank, mpi_shape)
        npt0 = [asn_dict[ax][m] for ax, m in zip(axes, coord)]
        npt1 = [asn_dict[ax][m+1] - 1 for ax, m in zip(axes, coord)]
        overlap = common.overlap_two_regions(npt0, npt1, pt0, pt1)

        if overlap != None:
            rank_list.append(rank)
            ox0, oy0, oz0 = overlap[0]
            ox1, oy1, oz1 = overlap[1]

            if is_first:
                ox, oy, oz = ox0, oy0, oz0
                is_first = False

            opt0 = (ox0-ox, oy0-oy, oz0-oz)
            opt1 = (ox1-ox, oy1-oy, oz1-oz)
            slices = common.slices_two_points(opt0, opt1)
            info_dict[rank] = [sl for sl in slices if sl != 0]

    info_dict['ranks'] = rank_list
    info_dict['shape'] = common.shape_two_points(pt0, pt1)
    info_dict['pt0'] = pt0
    info_dict['pt1'] = pt1
    info_dict['anx_list'] = asn_dict['dx']
    info_dict['any_list'] = asn_dict['y']
    info_dict['anz_list'] = asn_dict['z']

    return info_dict




if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #mpi_shape, pbc_axes = (2, 1, 1), ''
    #mpi_shape, pbc_axes = (2, 1, 1), 'xy'
    #mpi_shape, pbc_axes = (1, 2, 1), 'xyz'
    mpi_shape, pbc_axes = (2, 2, 1), 'xyz'
    #mpi_shape, pbc_axes = (3, 3, 1), 'xyz'

    if rank == 0: print mpi_shape, repr(pbc_axes)
    check_mpi_shape(size, mpi_shape)
    print 'rank= ', rank, mpi_target_dict(rank, mpi_shape, pbc_axes)

    if rank == 0:
        #asn_dict = accum_sub_ns_dict(mpi_shape, 2, [100, 10, 100, 10], [120], [130])
        asn_dict = accum_sub_ns_dict(mpi_shape, 2, [100, 10, 100, 10], [120, 140], [130])
        print 'accum_sub_ns_dict', asn_dict
        print 'divide_info_dict', divide_info_dict(4, (50, 50, 0), (149, 199, 0), asn_dict)

    print ''
