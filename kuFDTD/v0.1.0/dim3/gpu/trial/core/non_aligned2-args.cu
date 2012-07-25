__global__ void update_e( int Nz, int Nyz, int Nzm, int Nyzm, int Nxyzm, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, float *CEx, float *CEy, float *CEz ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int fidx = idx + idx/Nzm*2 + idx/Nyzm*Nz*2 + Nyz + Nz + 1; 
	
	if ( fidx < Nxyzm ) {
		Ex[fidx] += CEx[fidx]*( Hz[fidx+Nz] - Hz[fidx] - Hy[fidx+1] + Hy[fidx] );
		Ey[fidx] += CEy[fidx]*( Hx[fidx+1] - Hx[fidx] - Hz[fidx+Nyz] + Hz[fidx] );
		Ez[fidx] += CEz[fidx]*( Hy[fidx+Nyz] - Hy[fidx] - Hx[fidx+Nz] + Hx[fidx] );
	}
}


__global__ void update_h( int Nz, int Nyz, int Nzm, int Nyzm, int Nxyzm, float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int fidx = idx + idx/Nzm*2 + idx/Nyzm*Nz*2 + Nyz + Nz + 1; 

	if ( fidx < Nxyzm ) {
		Hx[fidx] -= 0.5*( Ez[fidx] - Ez[fidx-Nz] - Ey[fidx] + Ey[fidx-1] );
		Hy[fidx] -= 0.5*( Ex[fidx] - Ex[fidx-1] - Ez[fidx] + Ez[fidx-Nyz] );
		Hz[fidx] -= 0.5*( Ey[fidx] - Ey[fidx-Nyz] - Ex[fidx] + Ex[fidx-Nz] );
	}
}