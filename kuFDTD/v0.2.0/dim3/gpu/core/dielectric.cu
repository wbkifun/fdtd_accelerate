/*
 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Ki-Hwan Kim (wbkifun@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
*/

__global__ void update_e( int idx0, int Nz, int Nyz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk + idx0;
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
}


__global__ void update_h( int idx0, int Nz, int Nyz, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk + idx0;
	int eidx = idx + Nyz;
	
	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];
	float* ez = (float*) &ey[blockDim.x+1];

	ex[tk+1] = Ex[eidx];
	ey[tk+1] = Ey[eidx];
	ez[tk]   = Ez[eidx];
	if ( tk==0 ) {
		ex[0] = Ex[eidx-1];
		ey[0] = Ey[eidx-1];
	}
	__syncthreads();
	
	Hx[idx] -= 0.5*( ez[tk] - Ez[eidx-Nz] - ey[tk+1] + ey[tk] );
	Hy[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + Ez[eidx-Nyz] );
	Hz[idx] -= 0.5*( ey[tk+1] - Ey[eidx-Nyz] - ex[tk+1] + Ex[eidx-Nz] );
}
