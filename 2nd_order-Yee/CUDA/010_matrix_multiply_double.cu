/*------------------------------------------------------------------------------
# File Name : matrix_multiply_double.cu
#
# Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
# 
# Written date :	2010. 8. 17
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# CUDA example
# Matrix Multiplication C=AxB 
# This cuda kernel is not optimized.
------------------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>


__host__ void matrix_multiply_cpu(int n, double **a, double **b, double **c) {
	int i, j, k;
	double tc;
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			tc = 0;
			for (k=0; k<n; k++) tc += a[i][k]*b[k][j];
			c[i][j] = tc;
		}
	}
}


__global__ void matrix_multiply_gpu(int n, double *a, double *b, double *c) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	double tc = 0;
	for (int k=0; k<n; k++) tc += a[i*n+k]*b[k*n+j];
	c[i*n+j] = tc;
}


int main() {
	int n=512;	// n x n matrix; must be a multiple of 16
	int i,j;
	printf("Matrix Multiplication C=AxB (%dx%d) (double precision)\n", n,n);

	
	// Allocate matrices in the host memory
	double **a, **b, **c_cpu, **c_gpu;

	a = (double **) malloc(n*sizeof(double *));
	a[0] = (double *) malloc(n*n*sizeof(double));
	for (i=0; i<n; i++) a[i] = a[0] + i*n;

	b = (double **) malloc(n*sizeof(double *));
	b[0] = (double *) malloc(n*n*sizeof(double));
	for (i=0; i<n; i++) b[i] = b[0] + i*n;

	c_cpu = (double **) calloc(n, sizeof(double *));
	c_cpu[0] = (double *) calloc(n*n, sizeof(double));
	for (i=0; i<n; i++) c_cpu[i] = c_cpu[0] + i*n;

	c_gpu = (double **) calloc(n, sizeof(double *));
	c_gpu[0] = (double *) calloc(n*n, sizeof(double));
	for (i=0; i<n; i++) c_gpu[i] = c_gpu[0] + i*n;

	// Initialize the matrix a, b
	for (i=0; i<n*n; i++) {
		a[0][i] = (i/111)*(i%11)*0.1;
		b[0][i] = (i/113)*(i%13)*0.1;
	}

	// Allocate matrices in the device memory
	double *a_dev, *b_dev, *c_dev;
	cudaMalloc ( (void**) &a_dev, n*n*sizeof(double) );
	cudaMalloc ( (void**) &b_dev, n*n*sizeof(double) );
	cudaMalloc ( (void**) &c_dev, n*n*sizeof(double) );
	
	// Copy the a, b matrices from host to device
	cudaMemcpy ( a_dev, a[0], n*n*sizeof(double), cudaMemcpyHostToDevice );
	cudaMemcpy ( b_dev, b[0], n*n*sizeof(double), cudaMemcpyHostToDevice );
	cudaMemcpy ( c_dev, c_gpu[0], n*n*sizeof(double), cudaMemcpyHostToDevice );
	
	// CUDA Kernel execution
	dim3 dimBlock(16,16,1);
	dim3 dimGrid(n/16,n/16);
	printf("GPU Kernel Execution...\n");
	matrix_multiply_gpu <<<dimGrid,dimBlock>>> (n, a_dev, b_dev, c_dev);

	// Copy the c matrix from host to device
	printf("Copy the c matrix from host to device...\n");
	cudaMemcpy (c_gpu[0], c_dev, n*n*sizeof(double), cudaMemcpyDeviceToHost);


	// CPU Function execution
	printf("CPU Function Execution...\n");
	matrix_multiply_cpu(n, a, b, c_cpu);


	// Verify two results
	printf("Verify the results...");
	double v;
	int err=0;
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) {
			v = c_cpu[i][j] - c_gpu[i][j];
			if (v > 0.01) {
				printf("c[%d][%d]=%g\n", i, j, v);
				err = 1;
			}
		}
	}
	if (err == 0) printf("OK!\n");
}
