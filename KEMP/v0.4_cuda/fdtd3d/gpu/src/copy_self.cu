__global__ void copy_self(int nx, int ny, int nz, DTYPE *f) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx0, idx1;

	while( gid < NMAX ) {
        idx0 = IDX0;
        idx1 = IDX1;

	   	f[idx1] = f[idx0];
		gid += blockDim.x * gridDim.x;
	}
} 
