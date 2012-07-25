__global__ void update_e( int Nx, int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk;
	int Nyz = Ny*Nz;
	int i = idx/Nyz;
	int j = ( idx - i*Nyz )/Nz;
	int k = idx - i*Nyz - j*Nz;
	
	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[blockDim.x+1];
	float* hz = (float*) &hy[blockDim.x+1];

	if ( i > 0 && j > 0 && k > 0 && i < Nx ) {
		hx[tk] = Hx[idx];
		hy[tk] = Hy[idx];
		hz[tk] = Hz[idx];
		
		if ( tk==blockDim.x-1 ) {
			hx[tk+1] = Hx[idx+1];
			hy[tk+1] = Hy[idx+1];
		}
	}
	__syncthreads();
	
	if ( i > 0 && j > 0 && k > 0 && i < Nx ) {
		if ( j<Ny-1 && k<Nz-1 ) Ex[idx] += CEx[idx]*( Hz[idx+Nz] - hz[tk] - hy[tk+1] + hy[tk] );
		if ( i<Nx-1 && k<Nz-1 ) Ey[idx] += CEy[idx]*( hx[tk+1] - hx[tk] - Hz[idx+Nyz] + hz[tk] );
		if ( i<Nx-1 && j<Ny-1 ) Ez[idx] += CEz[idx]*( Hy[idx+Nyz] - hy[tk] - Hx[idx+Nz] + hx[tk] );
	}
}


__global__ void update_h( int Nx, int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk;
	int Nyz = Ny*Nz;
	int i = idx/Nyz;
	int j = ( idx - i*Nyz )/Nz;
	int k = idx - i*Nyz - j*Nz;
	
	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];
	float* ez = (float*) &ey[blockDim.x+1];

	if ( i > 0 && j > 0 && k > 0 && i < Nx ) {
		ex[tk+1] = Ex[idx];
		ey[tk+1] = Ey[idx];
		ez[tk] = Ez[idx];
		if ( tk==0 ) {
			ex[0] = Ex[idx-1];
			ey[0] = Ey[idx-1];
		}
	}
	__syncthreads();
	
	if ( i > 0 && j > 0 && k > 0 && i < Nx ) {
		Hx[idx] -= 0.5*( ez[tk] - Ez[idx-Nz] - ey[tk+1] + ey[tk] );
		Hy[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + Ez[idx-Nyz] );
		Hz[idx] -= 0.5*( ey[tk+1] - Ey[idx-Nyz] - ex[tk+1] + Ex[idx-Nz] );
	}
}
