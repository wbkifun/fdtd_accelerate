__global__ void update_e(int nx, int ny, int nz, DTYPE *ex, DTYPE *ey, DTYPE *ez, DTYPE *hx, DTYPE *hy, DTYPE *hz ARGS_CE) {
	int tx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tx;
	int i, j, k;
	
	__shared__ DTYPE sx[DX+1], sy[DX+1], sz[DX];

	//while( idx < nx*ny*nz ) {
    sx[tx] = hx[idx];
    sy[tx] = hy[idx];
    sz[tx] = hz[idx];
    if( tx == DX-1 ) {
        sx[tx+1] = hx[idx+1];
        sy[tx+1] = hy[idx+1];
    }
    __syncthreads();

    i = idx/(ny*nz);
    j = (idx - i*ny*nz)/nz;
    k = idx%nz;

    if( j<ny-1 && k<nz-1 PAD ) ex[idx] += CEX * ((hz[idx+nz] - sz[tx]) - (sy[tx+1] - sy[tx]));
    if( i<nx-1 && k<nz-1 PAD ) ey[idx] += CEY * ((sx[tx+1] - sx[tx]) - (hz[idx+ny*nz] - sz[tx]));
    if( i<nx-1 && j<ny-1 && k<nz PAD) ez[idx] += CEZ * ((hy[idx+ny*nz] - sy[tx]) - (hx[idx+nz] - sx[tx]));

		//idx += blockDim.x * gridDim.x;
        //__syncthreads();
	//}
}



__global__ void update_h(int nx, int ny, int nz, DTYPE *ex, DTYPE *ey, DTYPE *ez, DTYPE *hx, DTYPE *hy, DTYPE *hz ARGS_CH) {
	int tx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tx;
	int i, j, k;

	__shared__ DTYPE s[3*DX+2];
	__shared__ DTYPE *sx, *sy, *sz;
	sz = s;
	sy = &sz[DX+1];
	sx = &sy[DX+1];

	//while( idx < nx*ny*nz ) {
    sx[tx] = ex[idx];
    sy[tx] = ey[idx];
    sz[tx] = ez[idx];
    if( tx == 0 ) {
        sx[tx-1] = ex[idx-1];
        sy[tx-1] = ey[idx-1];
    }
    __syncthreads();

    i = idx/(ny*nz);
    j = (idx - i*ny*nz)/nz;
    k = idx%nz;

    if( j>0 && k>0 && k<nz PAD ) hx[idx] -= CHX * ((sz[tx] - ez[idx-nz]) - (sy[tx] - sy[tx-1]));
    if( i>0 && k>0 && k<nz PAD ) hy[idx] -= CHY * ((sx[tx] - sx[tx-1]) - (sz[tx] - ez[idx-ny*nz]));
    if( i>0 && j>0 && k<nz PAD ) hz[idx] -= CHZ * ((sy[tx] - ey[idx-ny*nz]) - (sx[tx] - ex[idx-nz]));

		//idx += blockDim.x * gridDim.x;
        //__syncthreads();
	//}
}
