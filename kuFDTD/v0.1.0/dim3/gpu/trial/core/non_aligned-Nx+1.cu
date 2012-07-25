__global__ void update_e( int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk;
	int Nyz = Ny*Nz;
	//int fidx = idx + idx/(Nz-1) + Nyz + Nz + 1; 
	int fidx = idx + idx/(Nz-1) + idx/( (Nz-1)*(Ny-1) )*(Nz-1) + Nyz + Nz + 1; 
	
	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[blockDim.x+1];
	float* hz = (float*) &hy[blockDim.x+1];

	hx[tk] = Hx[fidx];
	hy[tk] = Hy[fidx];
	hz[tk] = Hz[fidx];
	
	if ( tk==blockDim.x-1 ) {
		hx[tk+1] = Hx[fidx+1];
		hy[tk+1] = Hy[fidx+1];
	}
	__syncthreads();
	
	Ex[fidx] += CEx[fidx]*( Hz[fidx+Nz] - hz[tk] - hy[tk+1] + hy[tk] );
	Ey[fidx] += CEy[fidx]*( hx[tk+1] - hx[tk] - Hz[fidx+Nyz] + hz[tk] );
	Ez[fidx] += CEz[fidx]*( Hy[fidx+Nyz] - hy[tk] - Hx[fidx+Nz] + hx[tk] );
}


__global__ void update_h( int Ny, int Nz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk;
	int Nyz = Ny*Nz;
	//int fidx = idx + idx/(Nz-1) + Nyz + Nz + 1; 
	int fidx = idx + idx/(Nz-1) + idx/( (Nz-1)*(Ny-1) )*(Nz-1) + Nyz + Nz + 1; 
	
	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];
	float* ez = (float*) &ey[blockDim.x+1];

	ex[tk+1] = Ex[fidx];
	ey[tk+1] = Ey[fidx];
	ez[tk]   = Ez[fidx];
	if ( tk==0 ) {
		ex[0] = Ex[fidx-1];
		ey[0] = Ey[fidx-1];
	}
	__syncthreads();
	
	Hx[fidx] -= 0.5*( ez[tk] - Ez[fidx-Nz] - ey[tk+1] + ey[tk] );
	Hy[fidx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + Ez[fidx-Nyz] );
	Hz[fidx] -= 0.5*( ey[tk+1] - Ey[fidx-Nyz] - ex[tk+1] + Ex[fidx-Nz] );
}