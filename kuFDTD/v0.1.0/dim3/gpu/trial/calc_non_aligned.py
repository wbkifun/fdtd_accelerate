Nx, Ny, Nz = 10,10,10

for idx in xrange( (Nx-2)*(Ny-1)*(Nz-1) ):
	fidx = idx + idx/(Nz-1) + idx/( (Ny-1)*(Nz-1) )*Nz + Ny*Nz + Nz + 1

	print '%4.d, %4.d' % (idx, fidx)
