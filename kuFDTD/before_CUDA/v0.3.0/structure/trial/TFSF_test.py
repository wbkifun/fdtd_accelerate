#!/usr/bin/env python
#-*- coding: utf_8 -*-

"""
# x1
out_range[y][0] = ( i0, slice(j0 + 1, j1 + 2), slice(k0 + 1, k1 + 1) )
out_range[z][0] = ( i0, slice(j0 + 1, j1 + 1), slice(k0 + 1, k1 + 2) )

in_range[y][0] = ( i0, slice(j0    , j1    ), slice(k0    , k1 + 1) )
in_range[z][0] = ( i0, slice(j0    , j1 + 1), slice(k0    , k1    ) )
# x2
out_range[y][1] = ( i1 + 1, slice(j0 + 1, j1 + 2), slice(k0 + 1, k1 + 1) )
out_range[z][1] = ( i1 + 1, slice(j0 + 1, j1 + 1), slice(k0 + 1, k1 + 2) )

in_range[y][1] = ( i1    , slice(j0    , j1    ), slice(k0    , k1 + 1) )
in_range[z][1] = ( i1    , slice(j0    , j1 + 1), slice(k0    , k1    ) )

# y1
out_range[x][2] = ( slice(i0 + 1, i1 + 2), j0, slice(k0 + 1, k1 + 1) )
out_range[z][2] = ( slice(i0 + 1, i1 + 1), j0, slice(k0 + 1, k1 + 2) )
in_range[x][2] = ( slice(i0    , i1    ), j0, slice(k0    , k1 + 1) )
in_range[z][2] = ( slice(i0    , i1 + 1), j0, slice(k0    , k1    ) )
# y2
out_range[x][3] = ( slice(i0 + 1, i1 + 2), j1 + 1, slice(k0 + 1, k1 + 1) )
out_range[z][3] = ( slice(i0 + 1, i1 + 1), j1 + 1, slice(k0 + 1, k1 + 2) )
in_range[x][3] = ( slice(i0    , i1    ), j1    , slice(k0    , k1 + 1) )
in_range[z][3] = ( slice(i0    , i1 + 1), j1    , slice(k0    , k1    ) )

# z1
out_range[x][4] = ( slice(i0 + 1, i1 + 2), slice(j0 + 1, j1 + 1), k0 )
out_range[y][4] = ( slice(i0 + 1, i1 + 1), slice(j0 + 1, j1 + 2), k0 )
in_range[x][4] = ( slice(i0    , i1    ), slice(j0    , j1 + 1), k0) 
in_range[y][4] = ( slice(i0    , i1 + 1), slice(j0    , j1    ), k0) 
# z2
out_range[x][5] = ( slice(i0 + 1, i1 + 2), slice(j0 + 1, j1 + 1), k1 + 1 )
out_range[y][5] = ( slice(i0 + 1, i1 + 1), slice(j0 + 1, j1 + 2), k1 + 1 )
in_range[x][5] = ( slice(i0    , i1    ), slice(j0    , j1 + 1), k1    )
in_range[y][5] = ( slice(i0    , i1 + 1), slice(j0    , j1    ), k1    ) 
"""

def making_tfsf_slice(tfsf_range, dimension='3D'):
    # tfsf_range : [i1, i2, j1, j2, k1, k2]
    out_range = [[],[],[]] # outer field range --> outx, outy, outz
    in_range = [[],[],[]] # inner field range --> inx, iny, inz
    for i in xrange(3): # for 3 field
	for j in xrange(6): # for 6 direction : x1, x2, y1, y2, z1, z2
	    out_range[i].append(None)
	    in_range[i].append(None)
    
    di1 = tfsf_range[0::2] # [i1, j1, k1]  each direction front boundary index
    di2 = tfsf_range[1::2] # [i2, j2, k2]  each direction back boundary index
    for direc in xrange(6):
	axis0 = direc/2 # boundary direction
	axis1 = (((direc/2) - 1)/2)*-1 # field direction y(1), x(0), x(0)
	axis2 = (((direc/4) + 1)%2) + 1 # field direction z(2), z(2), y(1)
	out_range_temp1 = [ slice(di1[axis1] + 1, di2[axis1] + 2),\
			 	slice(di1[axis2] + 1, di2[axis2] + 1) ]
	out_range_temp2 = [ slice(di1[axis1] + 1, di2[axis1] + 1),\
				slice(di1[axis2] + 1, di2[axis2] + 2) ]
	in_range_temp1 = [ slice(di1[axis1]    , di2[axis1]    ),\
				slice(di1[axis2]    , di2[axis2] + 1) ]
	in_range_temp2 = [ slice(di1[axis1]    , di2[axis1] + 1),\
				slice(di1[axis2]    , di2[axis2]    ) ]
	out_range_temp1.insert(axis0, tfsf_range[direc] + direc%2)
	out_range_temp2.insert(axis0, tfsf_range[direc] + direc%2)
	in_range_temp1.insert(axis0, tfsf_range[direc])
	in_range_temp2.insert(axis0, tfsf_range[direc])

	out_range[axis1][direc] = tuple(out_range_temp1)
	out_range[axis2][direc] = tuple(out_range_temp2)
	in_range[axis1][direc] = tuple(in_range_temp1)
	in_range[axis2][direc] = tuple(in_range_temp2)
    if dimension == '2D':
	for i in xrange(3):
	    out_range[i] = out_range[i][:4]
	    in_range[i] = in_range[i][:4]
	    for j in xrange(4):
		if out_range[i][j] !=None:
		    out_range[i][j] = out_range[i][j][:2]
		if in_range[i][j] !=None:
		    in_range[i][j] = in_range[i][j][:2]

    return out_range, in_range

def making_tfsf_sign(dimension='3D'): 
    N = None
    efield_sign = [] # the updating field is efield
    hfield_sign = [] # the updating field is efield
    efield_sign.append( (N, N, -1, +1, +1, -1) )
    efield_sign.append( (+1, -1, N, N, -1, +1) )
    efield_sign.append( (-1, +1, +1, -1, N, N) )
    hfield_sign.append( (N, N, +1, -1, -1, +1) )
    hfield_sign.append( (-1, +1, N, N, +1, -1) )
    hfield_sign.append( (+1, -1, -1, +1, N, N) )
    if dimension == '2D':
	for i in xrange(3):
	    efield_sign[i] = efield_sign[i][:4]
	    hfield_sign[i] = hfield_sign[i][:4]

    return efield_sign, hfield_sign

def making_tfsf_polarization(dimension='3D'):
    x ,y, z = 0, 1, 2

    up_field1 = [y, y, x, x, x, x] 
    in_field1 = [z, z, z, z, y, y]
    up_field2 = [z, z, z, z, y, y]
    in_field2 = [y, y, x, x, x, x]
    if dimension == '2D':
	up_field1 = up_field1[:4] 
	in_field1 = in_field1[:4]
	up_field2 = up_field2[:4]
	in_field2 = in_field2[:4]
    up_field = [up_field1, up_field2]
    in_field = [in_field1, in_field2]
    
    return up_field, in_field

def update_tfsf_interface(update_field, incident_field, sign, cb):
	update_field += sign*cb*incident_field



if __name__ == '__main__':


    print
    print
    print '3D'
    out_range, in_range = making_tfsf_slice((0,7,0,2,0,2))
    for i in xrange(3):
	for j in xrange(6):
	    print '%s axis %s direction : Outer Field Range %s' % (i, j, out_range[i][j])
	    print '%s axis %s direction : Inner Field Range %s' % (i, j, in_range[i][j])
	    print
    efield_sign, hfield_sign = making_tfsf_sign()
    for i in xrange(3):
	for j in xrange(6):
	    print '%s axis %s direction : E Field Sign %s' % (i, j, efield_sign[i][j])
	    print '%s axis %s direction : H Field Sign %s' % (i, j, hfield_sign[i][j])
	    print

    print
    print
    print '2D'

    out_range, in_range = making_tfsf_slice((0,7,0,2,0,0),'2D')
    for i in xrange(3):
	for j in xrange(4):
	    print '%s axis %s direction : Outer Field Range %s' % (i, j, out_range[i][j])
	    print '%s axis %s direction : Inner Field Range %s' % (i, j, in_range[i][j])
	    print
    efield_sign, hfield_sign = making_tfsf_sign('2D')
    for i in xrange(3):
	for j in xrange(4):
	    print '%s axis %s direction : E Field Sign %s' % (i, j, efield_sign[i][j])
	    print '%s axis %s direction : H Field Sign %s' % (i, j, hfield_sign[i][j])
	    print



