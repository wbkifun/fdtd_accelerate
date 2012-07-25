__global__ void updateE( int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk;
	int Nyz = Ny*Nz;
	int eidx = idx + Nyz;
	
	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[blockDim.x+1];
	float* hz = (float*) &hy[blockDim.x+1];

	hx[tk] = Hx[idx];
	hy[tk] = Hy[idx];
	hz[tk] = Hz[idx];
	
	if ( tk==blockDim.x-1 ) {
		hx[tk+1] = Hx[idx+1];
		hy[tk+1] = Hy[idx+1];
	}
	__syncthreads();
	
	Ex[eidx] += CEx[idx]*( Hz[idx+Nz] - hz[tk] - hy[tk+1] + hy[tk] );
	Ey[eidx] += CEy[idx]*( hx[tk+1] - hx[tk] - Hz[idx+Nyz] + hz[tk] );
	Ez[eidx] += CEz[idx]*( Hy[idx+Nyz] - hy[tk] - Hx[idx+Nz] + hx[tk] );
	/*
	Ex[eidx] += CEx[idx]*( Hz[idx+Nz] - Hz[idx] - Hy[idx+1] + Hy[idx] );
	Ey[eidx] += CEy[idx]*( Hx[idx+1] - Hx[idx] - Hz[idx+Nyz] + Hz[idx] );
	Ez[eidx] += CEz[idx]*( Hy[idx+Nyz] - Hy[idx] - Hx[idx+Nz] + Hx[idx] );
	*/
}


__global__ void updateH( int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk;
	int Nyz = Ny*Nz;
	int eidx = idx + Nyz;
	
	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];
	float* ez = (float*) &ey[blockDim.x+1];

	ex[tk+1] = Ex[eidx];
	ey[tk+1] = Ey[eidx];
	ez[tk] = Ez[eidx];
	if ( tk==0 ) {
		ex[0] = Ex[eidx-1];
		ey[0] = Ey[eidx-1];
	}
	__syncthreads();
	
	Hx[idx] -= 0.5*( ez[tk] - Ez[eidx-Nz] - ey[tk+1] + ey[tk] );
	Hy[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + Ez[eidx-Nyz] );
	Hz[idx] -= 0.5*( ey[tk+1] - Ey[eidx-Nyz] - ex[tk+1] + Ex[eidx-Nz] );
	/*
	Hx[idx] -= 0.5*( Ez[eidx] - Ez[eidx-Nz] - Ey[eidx] + Ey[eidx-1] );
	Hy[idx] -= 0.5*( Ex[eidx] - Ex[eidx-1] - Ez[eidx] + Ez[eidx-Nyz] );
	Hz[idx] -= 0.5*( Ey[eidx] - Ey[eidx-Nyz] - Ex[eidx] + Ex[eidx-Nz] );
	*/
}
