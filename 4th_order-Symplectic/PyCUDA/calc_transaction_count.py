#!/usr/bin/env python

import scipy as sc

print 'Reduce the read access to the global memory per thread block'
print '(the thread block dimension and size)'

p = sc.array([2,4,6,8])
tn = 3 + 3 + 3*2*p
print '\nwithout using cache'
print tn, '(*n)'

print '\n2nd\t4th\t6th\t8th\t(spatial order)'
print '1D'  
for n,nx in [(256,256), (512,512)]:
	print '(%d=%d)' % (n,nx)
	n2 = 6*n + 3*n + (2*n+1) + (2*n+1)
	n4 = 6*n + 7*n + (4*n+3) + (4*n+3)
	n6 = 6*n + 11*n + (6*n+5) + (6*n+5)
	n8 = 6*n + 15*n + (8*n+7) + (8*n+7)
	#rn = 6.*n + (2*p-1)*n + (p*n+p-1) + (p*n+p-1)
	rn = 6.*n + (4*p-1)*n + 2*(p-1)
	#print rn
	ratio = rn/(tn*n)*100
	print sc.round_(ratio,2), '%'

print '\n2D'  
for n,nx,ny in [(256,16,16), (512,32,16)]:
	print '(%d=%dx%d)' % (n,nx,ny)
	n2 = 6*n + (2*n+nx) + (2*n+ny) + (n+(nx+ny))
	n4 = 6*n + (4*n+3*nx) + (4*n+3*ny) + (n+3*(nx+ny))
	#rn = 6.*n + (p*n+(p-1)*nx) + (p*n+(p-1)*ny) + (n+(p-1)*(nx+ny))
	rn = 6.*n + (2*p+1)*n + 2*(p-1)*(nx+ny)
	ratio = rn/(tn*n)*100
	print sc.round_(ratio,2), '%'

print '\n3D'  
for n,nx,ny,nz in [(256,16,4,4), (512,16,8,4)]:
	print '(%d=%dx%dx%d)' % (n,nx,ny,nz)
	n2 = 6*n + 3*n + 2*(nx*ny+ny*nz+nz*nx)
	n3 = 6*n + 3*n + 6*(nx*ny+ny*nz+nz*nx)
	rn = 6.*n + 3*n + (p-1)*2*(nx*ny+ny*nz+nz*nx)
	ratio = rn/(tn*n)*100
	print sc.round_(ratio,2), '%'


# transaction count
tn = 6 + 7*p-1
print '\n\nTransaction count'  
print '1D'  
for n,nx in [(256,256), (512,512)]:
	print '(%d=%d)' % (n,nx)
	nc = n/16
	rn = 6.*nc + (4*p-1)*nc + sc.array([2,4,4,4])
	ratio = rn/(tn*nc)*100
	print sc.round_(ratio,2), '%'

print '\n2D'  
for n,nx,ny in [(256,16,16), (512,32,16)]:
	print '(%d=%dx%d)' % (n,nx,ny)
	nc = n/16
	rn = 6.*nc + (2*p+1)*nc + 2*(p-1)*(nx/16) + sc.array([2,4,4,4])*ny
	ratio = rn/(tn*nc)*100
	print sc.round_(ratio,2), '%'

print '\n3D'  
for n,nx,ny,nz in [(256,16,4,4), (512,16,8,4)]:
	print '(%d=%dx%dx%d)' % (n,nx,ny,nz)
	nc = n/16
	rn = 6.*nc + 3*nc + 2*(p-1)*(nx/16)*(ny+nz) + sc.array([2,4,4,4])*ny*nz
	ratio = rn/(tn*nc)*100
	print sc.round_(ratio,2), '%'


# transaction count of applied mis-aligned access with texture fetch
print '\n\nTransaction count of applied mis-aligned access with texture fetch'  
print '1D'  
print 'same above'

print '\n2D'  
for n,nx,ny in [(256,16,16), (512,32,16)]:
	print '(%d=%dx%d)' % (n,nx,ny)
	nc = n/16
	rn = 6.*nc + (2*p+1)*nc + 2*(p-1)*(nx/16) + 2*ny
	ratio = rn/(tn*nc)*100
	print sc.round_(ratio,2), '%'

print '\n3D'  
for n,nx,ny,nz in [(256,16,4,4), (512,16,8,4)]:
	print '(%d=%dx%dx%d)' % (n,nx,ny,nz)
	nc = n/16
	rn = 6.*nc + 3*nc + 2*(p-1)*(nx/16)*(ny+nz) + 2*ny*nz
	ratio = rn/(tn*nc)*100
	print sc.round_(ratio,2), '%'

# transaction count for using rotation
print '\n\nTransaction count for using rotation'  
print '3D'  
for n,nx,ny,nz in [(256,16,4,4), (512,16,8,4)]:
	print '(%d=%dx%dx%d)' % (n,nx,ny,nz)
	nc = n/16
	rn = 6.*nc + 3*nc + 3*(p-1)*(nx/16)*(ny+nz)
	ratio = rn/(tn*nc)*100
	print sc.round_(ratio,2), '%'
