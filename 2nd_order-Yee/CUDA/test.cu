#include <stdio.h>
#include <cutil.h>

__global__ void kern(int n, float *a, float *b) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	while( idx < n )
		b[idx] = 0.5*a[idx];
		idx += blockDim.x * gridDim.x;
}

int main() {
	//int n=1024;
	int n=100000000;

	float *a_gpu, *b_gpu;
	cudaMalloc( (void**) &a_gpu, n*sizeof(float) );
	cudaMalloc( (void**) &b_gpu, n*sizeof(float) );

	//kern <<<n/256,256>>> (n, a_gpu, b_gpu);
	kern <<<65535,256>>> (n, a_gpu, b_gpu);
	return 0;
}
