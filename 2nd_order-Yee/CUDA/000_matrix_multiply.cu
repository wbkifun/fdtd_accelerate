#include <stdlib.h>
#include <stdio.h>


__host__ void matrix_multiply_cpu(int n, float **a, float **b, float **c) {
	int i, j, k;
	float tc;
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			tc = 0;
			for (k=0; k<n; k++) tc += a[i][k]*b[k][j];
			c[i][j] = tc;
		}
	}
}


__global__ void matrix_multiply_gpu(int n, float *a, float *b, float *c) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	float tc = 0;
	for (int k=0; k<n; k++) tc += a[i*n+k]*b[k*n+j];
	c[i*n+j] = tc;
}


int main() {
	int n=256;	// n x n matrix; must be a multiple of 16
	int i,j;

	// Allocate matrices in the host memory
	float **a, **b, **c_cpu, **c_gpu;

	a = (float **) malloc(n*sizeof(float *));
	a[0] = (float *) malloc(n*n*sizeof(float));
	for (i=0; i<n; i++) a[i] = a[0] + i*n;

	b = (float **) malloc(n*sizeof(float *));
	b[0] = (float *) malloc(n*n*sizeof(float));
	for (i=0; i<n; i++) b[i] = b[0] + i*n;

	c_cpu = (float **) calloc(n, sizeof(float *));
	c_cpu[0] = (float *) calloc(n*n, sizeof(float));
	for (i=0; i<n; i++) c_cpu[i] = c_cpu[0] + i*n;

	c_gpu = (float **) calloc(n, sizeof(float *));
	c_gpu[0] = (float *) calloc(n*n, sizeof(float));
	for (i=0; i<n; i++) c_gpu[i] = c_gpu[0] + i*n;

	// Initialize the matrix a, b
	for (i=0; i<n*n; i++) {
		a[0][i] = (i/111)*(i%11)*0.1;
		b[0][i] = (i/113)*(i%13)*0.1;
	}

	// Allocate matrices in the device memory
	float *a_dev, *b_dev, *c_dev;
	cudaMalloc ( (void**) &a_dev, n*n*sizeof(float) );
	cudaMalloc ( (void**) &b_dev, n*n*sizeof(float) );
	cudaMalloc ( (void**) &c_dev, n*n*sizeof(float) );
	
	// Copy the a, b matrices from host to device
	cudaMemcpy ( a_dev, a[0], n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy ( b_dev, b[0], n*n*sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy ( c_dev, c_gpu[0], n*n*sizeof(float), cudaMemcpyHostToDevice );
	
	// CUDA Kernel execution
	dim3 dimBlock(16,16,1);
	dim3 dimGrid(n/16,n/16);
	matrix_multiply_gpu <<<dimGrid,dimBlock>>> (n, a_dev, b_dev, c_dev);

	// Copy the a, b matrices from host to device
	cudaMemcpy (c_gpu[0], c_dev, n*n*sizeof(float), cudaMemcpyDeviceToHost);


	// CPU Function execution
	matrix_multiply_cpu(n, a, b, c_cpu);


	// Verify two results
	float v;
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			v = c_cpu[i][j] - c_gpu[i][j];
			if (v > 0.01) printf("c[%d][%d]=%g\n", i, j, v);
		}
	}
}
