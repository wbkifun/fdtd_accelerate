#include <stdlib.h>
#include <stdio.h>


__global__ void kern(float *a) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	a[idx] = a[idx] + 3.0;
}


int main() {
	int tn, nx=32*16*2;
	
	float *a, *b, *c;
	a = (float *) malloc (nx*sizeof(float));
	b = (float *) malloc (nx*sizeof(float));
	c = (float *) malloc (nx*sizeof(float));

	float *a_gpu, *b_gpu, *c_gpu;
	int size = nx*sizeof(float);
	cudaMalloc ( (void**) &a_gpu, size );
	cudaMalloc ( (void**) &b_gpu, size );
	cudaMalloc ( (void**) &c_gpu, size );

	cudaMemcpy ( a_gpu, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy ( b_gpu, b, size, cudaMemcpyHostToDevice );
	cudaMemcpy ( c_gpu, c, size, cudaMemcpyHostToDevice );
	
	int tpb = 512;
	int bpg = nx/tpb;

	for(tn=1; tn<=10; tn++) {
		kern <<<dim3(bpg),dim3(tpb)>>> (a_gpu);
	}
}
