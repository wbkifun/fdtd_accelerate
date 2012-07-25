#include <stdio.h>
#include <stdlib.h>

__global__ void update_h(int nx, int ny, int nz, int nyz, int idx0, float *hx, float *hy, float *hz, float *ex, float *ey, float *ez) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx + idx0;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j>0 && k>0 ) hx[idx] -= 0.5*( ez[idx] - ez[idx-nz] - ey[idx] + ey[idx-1] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( ex[idx] - ex[idx-1] - ez[idx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( ey[idx] - ey[idx-nyz] - ex[idx] + ex[idx-nz] );
}

__global__ void update_e(int nx, int ny, int nz, int nyz, int idx0, float *hx, float *hy, float *hz, float *ex, float *ey, float *ez, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx + idx0;
	int i = idx/(nyz);
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+nyz] + hz[idx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - hy[idx] - hx[idx+nz] + hx[idx] );
}

__global__ void update_src(int nx, int ny, int nz, int nyz, float tn, float *f) {
	int idx = threadIdx.x;
	int ijk = (nx/2)*nyz + (ny/2)*nz + idx;

	if( idx < nz ) f[ijk] += sin(0.1*tn);
}
  
__global__ void init_zero(int n, int idx0, float *f) {
	int tx = threadIdx.x;
	int idx = blockDim.x*blockIdx.x + tx + idx0;
	if( idx < n ) f[idx] = 0;
}


int main() {
	int i, n, nx, ny, nz, tn, tmax;
	nx = 320;
	ny = 480;
	nz = 480;
	n = nx*ny*nz;
	tmax = 100000;

	printf("Simple FDTD simulation\n", nx, ny, nz);
	printf("Array size : %dx%dx%d\n", nx, ny, nz);
	printf("Total used memory : %1.2f GB\n", n*4*9./(1024*1024*1024));
	printf("Iteration : %d step\n", tmax);

	// memory allocate
	float *f, *cf;
	f = (float *) calloc (n, sizeof(float));
	cf = (float *) calloc (n, sizeof(float));
	for( i=0; i<n; i++ ) cf[i] = 0.5;

	float *hx_gpu, *hy_gpu, *hz_gpu;
	float *ex_gpu, *ey_gpu, *ez_gpu;
	float *cex_gpu, *cey_gpu, *cez_gpu;
	cudaMalloc ( (void**) &hx_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &hy_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &hz_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &ex_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &ey_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &ez_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &cex_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &cey_gpu, n*sizeof(float) );
	cudaMalloc ( (void**) &cez_gpu, n*sizeof(float) );

	cudaMemcpy ( cex_gpu, cf, n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy ( cey_gpu, cf, n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy ( cez_gpu, cf, n*sizeof(float), cudaMemcpyHostToDevice );

	int ng = 6;			// number of grid
	int tpb = 256;		// threads per block
	int bpg = n/tpb/ng;	// blocks per grid
	for( i=0; i<ng; i++) {
		init_zero <<<dim3(bpg),dim3(tpb)>>> (n, i*bpg*tpb, hx_gpu);
		init_zero <<<dim3(bpg),dim3(tpb)>>> (n, i*bpg*tpb, hy_gpu);
		init_zero <<<dim3(bpg),dim3(tpb)>>> (n, i*bpg*tpb, hz_gpu);
		init_zero <<<dim3(bpg),dim3(tpb)>>> (n, i*bpg*tpb, ex_gpu);
		init_zero <<<dim3(bpg),dim3(tpb)>>> (n, i*bpg*tpb, ey_gpu);
		init_zero <<<dim3(bpg),dim3(tpb)>>> (n, i*bpg*tpb, ez_gpu);
	}

	// main loop
	for( tn=0; tn<tmax; tn++ ) {
		for( i=0; i<ng; i++) update_h <<<dim3(bpg),dim3(tpb)>>> (nx, ny, nz, ny*nz, i*bpg*tpb, hx_gpu, hy_gpu, hz_gpu, ex_gpu, ey_gpu, ez_gpu); 

		for( i=0; i<ng; i++) update_e <<<dim3(bpg),dim3(tpb)>>> (nx, ny, nz, ny*nz, i*bpg*tpb, hx_gpu, hy_gpu, hz_gpu, ex_gpu, ey_gpu, ez_gpu, cex_gpu, cey_gpu, cez_gpu); 

		update_src <<<dim3(1),dim3(512)>>> (nx, ny, nz, ny*nz, tn, ez_gpu);
	}
	cudaMemcpy( f, ez_gpu, n*sizeof(float), cudaMemcpyDeviceToHost );
	printf("Complete.\n");

	return 0;
}
