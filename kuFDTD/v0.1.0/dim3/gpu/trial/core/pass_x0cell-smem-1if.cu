__global__ void update_e( int Ntot, int Nz, int Nyz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk + Nyz;
	
	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[blockDim.x+1];
	float* hz = (float*) &hy[blockDim.x+1];

	if ( idx < Ntot ) {
		hx[tk] = Hx[idx];
		hy[tk] = Hy[idx];
		hz[tk] = Hz[idx];
		
		if ( tk==blockDim.x-1 ) {
			hx[tk+1] = Hx[idx+1];
			hy[tk+1] = Hy[idx+1];
		}
	}
	__syncthreads();
	
	if ( idx < Ntot ) {
		Ex[idx] += CEx[idx]*( Hz[idx+Nz] - hz[tk] - hy[tk+1] + hy[tk] );
		Ey[idx] += CEy[idx]*( hx[tk+1] - hx[tk] - Hz[idx+Nyz] + hz[tk] );
		Ez[idx] += CEz[idx]*( Hy[idx+Nyz] - hy[tk] - Hx[idx+Nz] + hx[tk] );
	}
}


__global__ void update_h( int Ntot, int Nz, int Nyz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk + Nyz;
	
	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];
	float* ez = (float*) &ey[blockDim.x+1];

	if ( idx < Ntot ) {
		ex[tk+1] = Ex[idx];
		ey[tk+1] = Ey[idx];
		ez[tk]   = Ez[idx];
		if ( tk==0 ) {
			ex[0] = Ex[idx-1];
			ey[0] = Ey[idx-1];
		}
	}
	__syncthreads();
	
	if ( idx < Ntot ) {
		Hx[idx] -= 0.5*( ez[tk] - Ez[idx-Nz] - ey[tk+1] + ey[tk] );
		Hy[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + Ez[idx-Nyz] );
		Hz[idx] -= 0.5*( ey[tk+1] - Ey[idx-Nyz] - ex[tk+1] + Ex[idx-Nz] );
	}
}
