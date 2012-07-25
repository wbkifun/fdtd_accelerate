#!/usr/bin/env python


def update_h(ex, ey, ez, hx, hy, hz):
	hx[:,1:,1:] -= 0.5*(ez[:,1:,1:] - ez[:,:-1,1:] - ey[:,1:,1:] + ey[:,1:,:-1])
	hy[1:,:,1:] -= 0.5*(ex[1:,:,1:] - ex[1:,:,:-1] - ez[1:,:,1:] + ez[:-1,:,1:])
	hz[1:,1:,:] -= 0.5*(ey[1:,1:,:] - ey[:-1,1:,:] - ex[1:,1:,:] + ex[1:,:-1,:])


def update_e(ex, ey, ez, hx, hy, hz, cex, cey, cez):
	ex[:,:-1,:-1] += cex[:,:-1,:-1]*(hz[:,1:,:-1] - hz[:,:-1,:-1] - hy[:,:-1,1:] + hy[:,:-1,:-1])
	ey[:-1,:,:-1] += cey[:-1,:,:-1]*(hx[:-1,:,1:] - hx[:-1,:,:-1] - hz[1:,:,:-1] + hz[:-1,:,:-1])
	ez[:-1,:-1,:] += cez[:-1,:-1,:]*(hy[1:,:-1,:] - hy[:-1,:-1,:] - hx[:-1,1:,:] + hx[:-1,:-1,:])


def update_pbc_h(directions, hx, hy, hz):
	if 'x' in directions:
		hy[0,:,:] = hy[-1,:,:]
		hz[0,:,:] = hz[-1,:,:]
	if 'y' in directions:
		hz[:,0,:] = hz[:,-1,:]
		hx[:,0,:] = hx[:,-1,:]
	if 'z' in directions:
		hx[:,:,0] = hx[:,:,-1]
		hy[:,:,0] = hy[:,:,-1]
	

def update_pbc_e(directions, ex, ey, ez):
	if 'x' in directions:
		ey[-1,:,:] = ey[0,:,:]
		ez[-1,:,:] = ez[0,:,:]
	if 'y' in directions:
		ez[:,-1,:] = ez[:,0,:]
		ex[:,-1,:] = ex[:,0,:]
	if 'z' in directions:
		ex[:,:,-1] = ex[:,:,0]
		ey[:,:,-1] = ey[:,:,0]
