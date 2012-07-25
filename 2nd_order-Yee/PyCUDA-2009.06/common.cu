__global__ void initArray( int Ntot, float *a ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ( idx < Ntot ) a[idx] = 0;
}
