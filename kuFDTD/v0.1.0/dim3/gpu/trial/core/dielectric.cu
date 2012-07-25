__global__ void update_e( int Nx, int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int Nyz = Ny*Nz;
	int i = idx/Nyz;
	int j = ( idx - i*Nyz )/Nz;
	int k = idx - i*Nyz - j*Nz;
	
	if ( i > 0 && j > 0 && k > 0 && i < Nx ) {
		if ( j<Ny-1 && k<Nz-1 ) Ex[idx] += CEx[idx]*( Hz[idx+Nz] - Hz[idx] - Hy[idx+1] + Hy[idx] );
		if ( i<Nx-1 && k<Nz-1 ) Ey[idx] += CEy[idx]*( Hx[idx+1] - Hx[idx] - Hz[idx+Nyz] + Hz[idx] );
		if ( i<Nx-1 && j<Ny-1 ) Ez[idx] += CEz[idx]*( Hy[idx+Nyz] - Hy[idx] - Hx[idx+Nz] + Hx[idx] );
	}
}


__global__ void update_h( int Nx, int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int Nyz = Ny*Nz;
	int i = idx/Nyz;
	int j = ( idx - i*Nyz )/Nz;
	int k = idx - i*Nyz - j*Nz;
	
	if ( i > 0 && j > 0 && k > 0 && i < Nx ) {
		Hx[idx] -= 0.5*( Ez[idx] - Ez[idx-Nz] - Ey[idx] + Ey[idx-1] );
		Hy[idx] -= 0.5*( Ex[idx] - Ex[idx-1] - Ez[idx] + Ez[idx-Nyz] );
		Hz[idx] -= 0.5*( Ey[idx] - Ey[idx-Nyz] - Ex[idx] + Ex[idx-Nz] );
	}
}
