#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>


__host__ void updateTimer(time_t t0, int tstep, char str[]) {
	int elapsedTime=(int)(time(0)-t0);
	sprintf(str, "%02d:%02d:%02d", elapsedTime/3600, elapsedTime%3600/60, elapsedTime%60);
}


__host__ void exec(char *format, ...) {
	char str[1024];
	va_list ap;
	va_start(ap, format);
	vsprintf(str, format, ap);
	system(str);
}


__host__ void dumpToH5(int Ni, int Nj, int Nk, int is, int js, int ks, int ie, int je, int ke, float ***f, char *format, ...) {
	char filename[1024];
	va_list ap;
	va_start(ap, format);
	vsprintf(filename, format, ap);
	hid_t file, dataset, filespace, memspace;

	hsize_t dimsm[3] = { Ni, Nj, Nk };
	hsize_t start[3] = { is, js, ks };
	hsize_t count[3] = { 1-is+ie, 1-js+je, 1-ks+ke };
	memspace = H5Screate_simple(3, dimsm, 0);
	filespace = H5Screate_simple(3, count, 0);
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	dataset = H5Dcreate(file, "Data", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT);
	H5Sselect_hyperslab(memspace, H5S_SELECT_SET, start, 0, count, 0);
	H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, f[0][0]);
	H5Dclose(dataset);
	H5Sclose(filespace);
	H5Sclose(memspace);
	H5Fclose(file);
}


__host__ float ***makeArray(int Nx, int Ny, int Nz) {
	float ***f;

	f = (float ***) calloc (Nx, sizeof(float **));
	f[0] = (float **) calloc (Ny*Nx, sizeof(float *));
	f[0][0] = (float *) calloc (Nz*Ny*Nx, sizeof(float));

	for (int i=0; i<Nx; i++) f[i] = f[0] + i*Ny;
	for (int i=0; i<Ny*Nx; i++) f[0][i] = f[0][0] + i*Nz;

	return f;
}


__host__ void set_geometry(int Nx, int Ny, int Nz,
		float ***CEx, float ***CEy, float ***CEz) {
	int i,j,k;

	for (i=0; i<Nx-1; i++) {
		for (j=0; j<Ny-1; j++) {
			for (k=0; k<Nz-1; k++) {
				CEx[i][j][k] = 0.5;
				CEy[i][j][k] = 0.5;
				CEz[i][j][k] = 0.5;
			}
		}
	}
}


__global__ void initArrays(int Nx, int Ny, int Nz, 
		float *Ex, float *Ey, float *Ez, 
		float *Hx, float *Hy, float *Hz) {
	int idx;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( idx < Nx*Ny*Nz ) {
		Ex[idx] = 0;
		Ey[idx] = 0;
		Ez[idx] = 0;
		Hx[idx] = 0;
		Hy[idx] = 0;
		Hz[idx] = 0;
	}
}


__global__ void updateE(int Nx, int Ny, int Nz,
		float *Ex, float *Ey, float *Ez, 
		float *Hx, float *Hy, float *Hz, 
		float *CEx, float *CEy, float *CEz) {
	int idx,i,j,k;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	int Nyz = Ny*Nz;
	i = idx/Nyz;
	j = ( idx - i*Nyz )/Nz;
	k = idx - i*Nyz - j*Nz;

	if ( idx < Nx*Ny*Nz ) {
		if ( j<Ny-1 && k<Nz-1 ) Ex[idx] += CEx[idx]*( Hz[idx+Nz] - Hz[idx] - Hy[idx+1] + Hy[idx] );
		if ( i<Nx-1 && k<Nz-1 ) Ey[idx] += CEy[idx]*( Hx[idx+1] - Hx[idx] - Hz[idx+Nz*Ny] + Hz[idx] );
		if ( i<Nx-1 && j<Ny-1 ) Ez[idx] += CEz[idx]*( Hy[idx+Nz*Ny] - Hy[idx] - Hx[idx+Nz] + Hx[idx] );
	}
}


__global__ void updateSrc(int Nx, int Ny, int Nz,
		float *Ex, int tstep) {
	int idx, ijk;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	ijk = idx*(Ny)*(Nz) + (Ny/2)*(Nz) + (Nz)/2;

	//printf("idx=%d, ijk=%d\n", idx, ijk);
	//Ex[ijk] += __sinf(0.1*tstep);
	if ( idx < Nx ) {
		Ex[ijk] += sin(0.1*tstep);
	}
}


__global__ void updateH(int Nx, int Ny, int Nz,
		float *Ex, float *Ey, float *Ez, 
		float *Hx, float *Hy, float *Hz) {
	int idx, i,j,k;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	int Nyz = Ny*Nz;
	i = idx/Nyz;
	j = ( idx - i*Nyz )/Nz;
	k = idx - i*Nyz - j*Nz;

	if ( idx < Nx*Ny*Nz ) {
		if ( j>0 && k>0 ) Hx[idx] -= 0.5*( Ez[idx] - Ez[idx-Nz] - Ey[idx] + Ey[idx-1] );
		if ( i>0 && k>0 ) Hy[idx] -= 0.5*( Ex[idx] - Ex[idx-1] - Ez[idx] + Ez[idx-Nyz] );
		if ( i>0 && j>0 ) Hz[idx] -= 0.5*( Ey[idx] - Ey[idx-Nyz] - Ex[idx] + Ex[idx-Nz] );
	}
}


int main() {
	int tstep;
	char time_str[32];
	time_t t0;

	// Set the parameters
	int Nx, Ny, Nz, TMAX;
	Nx = 100;
	Ny = 200;
	Nz = 500;
	TMAX = 1000;

	// Allocate host memory
	//float ***Ex;
	float ***CEx, ***CEy, ***CEz;
	//Ex = makeArray(Nx, Ny, Nz);
	CEx = makeArray(Nx, Ny, Nz);
	CEy = makeArray(Nx, Ny, Nz);
	CEz = makeArray(Nx, Ny, Nz);

	// Geometry
	set_geometry(Nx, Ny, Nz, CEx, CEy, CEz);


	// Allocate device memory
	float *devEx, *devEy, *devEz;
	float *devHx, *devHy, *devHz;
	float *devCEx, *devCEy, *devCEz;
	int array_size = Nx*Ny*Nz*sizeof(float);
	cudaMalloc ( (void**) &devEx, array_size );
	cudaMalloc ( (void**) &devEy, array_size );
	cudaMalloc ( (void**) &devEz, array_size );
	cudaMalloc ( (void**) &devHx, array_size );
	cudaMalloc ( (void**) &devHy, array_size );
	cudaMalloc ( (void**) &devHz, array_size );
	cudaMalloc ( (void**) &devCEx, array_size );
	cudaMalloc ( (void**) &devCEy, array_size );
	cudaMalloc ( (void**) &devCEz, array_size );
	
	// Copy arrays from host to device
	cudaMemcpy ( devCEx, CEx[0][0], array_size, cudaMemcpyHostToDevice );
	cudaMemcpy ( devCEy, CEy[0][0], array_size, cudaMemcpyHostToDevice );
	cudaMemcpy ( devCEz, CEz[0][0], array_size, cudaMemcpyHostToDevice );
	
	// Number of thread blocks in the grid
	int N = Nx*Ny*Nz;
	int TPB = 256;
	int BPG = N%TPB == 0 ? N/TPB : N/TPB + 1;
	printf("BPG = %d\n", BPG);
	dim3 gridDim(BPG);
	// Number of threads per block
	dim3 blockDim(TPB);

	int BPGsrc = Nx%TPB == 0 ? Nx/TPB : Nx/TPB + 1;
	dim3 gridDimsrc(BPGsrc);
	dim3 blockDimsrc(Nx);

	// Initialize the device arrays
	initArrays <<<gridDim,blockDim>>> ( Nx, Ny, Nz, devEx, devEy, devEz, devHx, devHy, devHz );

	// Main time loop
	t0 = time(0);
	//for ( tstep=1; tstep<=TMAX; tstep++) {
	for ( tstep=1; tstep<=10; tstep++) {
		// Update on the GPU
		updateE <<<gridDim,blockDim>>> (Nx, Ny, Nz, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz);
		//updateSrc <<<gridDimsrc,blockDimsrc>>> (Nx, Ny, Nz, devEx, tstep);
		//updateH <<<gridDim,blockDim>>> (Nx, Ny, Nz, devEx, devEy, devEz, devHx, devHy, devHz);

		//print_array(dev2Ex);
		
		/*
		//if ( tstep/10*10 == tstep ) {
			// Copy arrays from device to host
			cudaMemcpy (Ex[0][0], devEx, array_size, cudaMemcpyDeviceToHost);

			dumpToH5(Nx, Ny, Nz, Nx/2, 0, 0, Nx/2, Ny-1, Nz-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		//}
		*/
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
}
