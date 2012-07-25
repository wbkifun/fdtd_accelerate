#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

#define TPBx 16	// Number of threads per block
#define TPBy 16


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


__host__ void print_array(int Nx, int Ny, int Nz, float ***a) {
	int j,k;
	for (j=0; j<Ny; j++) {
		for (k=0; k<Nz; k++) {
			printf("%1.4f\t", a[Nx/2][j][k]);
		}
		printf("\n");
	}
	printf("\n");
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

	for (i=0; i<Nx; i++) {
		for (j=0; j<Ny; j++) {
			for (k=0; k<Nz; k++) {
				CEx[i][j][k] = 0.5;
				CEy[i][j][k] = 0.5;
				CEz[i][j][k] = 0.5;
			}
		}
	}
}


__global__ void initArrays(int Nx, int Ny, int Nzpit,
		float *Ex, float *Ey, float *Ez, 
		float *Hx, float *Hy, float *Hz) {
	int idx;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	//printf("gridDim.x=%d\n",gridDim.x);
	//printf("blockIdx.x=%d, blockDim.x=%d, threadIdx.x=%d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if ( idx < Nx*Ny*Nzpit ) {
		Ex[idx] = 0;
		Ey[idx] = 0;
		Ez[idx] = 0;
		Hx[idx] = 0;
		Hy[idx] = 0;
		Hz[idx] = 0;
	}
}


__global__ void updateE(int Nx, int Ny, int Nz, int Nzpit, int BPGy,
		float *Ex, float *Ey, float *Ez, 
		float *Hx, float *Hy, float *Hz, 
		float *CEx, float *CEy, float *CEz) {
	int tk, tj;
	int i, j, k;
	tk = threadIdx.x;
	tj = threadIdx.y;
	k = TPBx*blockIdx.x + tk;
	j = TPBy*(blockIdx.y % BPGy ) + tj;
	i = blockIdx.y / BPGy;

	//printf("[%d, %d] [%d, %d, %d]\n", tk, tj, k, j, i);
	//printf("gridDim.x=%d, gridDim.y=%d\n", gridDim.x, gridDim.y );
	printf("blockIdx.x=%d, blockIdx.y=%d\n", blockIdx.x, blockIdx.y );
	//printf("BPGy=%d, blockIdx.y=%d, %BPGy=%d, /BPGy=%d\n", BPGy, blockIdx.y, blockIdx.y%BPGy, blockIdx.y/BPGy );

	if ( i<Nx && j<Ny && k<Nz ) {
		int Nyzpit = Ny*Nzpit;
		int idx = k + Nzpit*j + Nyzpit*i;

		//printf("idx=%d, [%d, %d] [%d, %d, %d]\n", idx, tk, tj, k, j, i);
		//printf("idx=%d\n", idx);
		__shared__ float hx[TPBy+1][TPBx+1], hy[TPBy][TPBx+1], hz[TPBy+1][TPBx];
		hx[tj][tk] = Hx[idx];
		hy[tj][tk] = Hy[idx];
		hz[tj][tk] = Hz[idx];
		if ( tk==TPBx-1 && k<Nz-1 ) {
			hx[tj][tk+1] = Hx[idx+1];
			hy[tj][tk+1] = Hy[idx+1];
		}
		if ( tj==TPBy-1 && j<Ny-1 ) {
			hx[tj+1][tk] = Hx[idx+Nzpit];
			hz[tj+1][tk] = Hz[idx+Nzpit];
		}
		__syncthreads();

		if ( k < Nz ) {
			if ( j<Ny-1 && k<Nz-1 ) Ex[idx] += CEx[idx]*( hz[tj+1][tk] - hz[tj][tk] - hy[tj][tk+1] + hy[tj][tk] );
			if ( i<Nx-1 && k<Nz-1 ) Ey[idx] += CEy[idx]*( hx[tj][tk+1] - hx[tj][tk] - Hz[idx+Nyzpit] + hz[tj][tk] );
			if ( i<Nx-1 && j<Ny-1 ) Ez[idx] += CEz[idx]*( Hy[idx+Nyzpit] - hy[tj][tk] - hx[tj+1][tk] + hx[tj][tk] );
		}
	}
}


__global__ void updateSrc(int Nx, int Ny, int Nz, int Nzpit,
	   	float *Ex, int tstep) {
	int idx, ijk;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	ijk = idx*(Ny)*(Nzpit) + (Ny/2)*(Nzpit) + (Nz/2);

	//printf("idx=%d, ijk=%d\n", idx, ijk);
	//Ex[ijk] += __sinf(0.1*tstep);
	if ( idx < Nx ) {
		Ex[ijk] += sin(0.1*tstep);
	}
}


__global__ void updateH(int Nx, int Ny, int Nz, int Nzpit, int BPGy,
		float *Ex, float *Ey, float *Ez, 
		float *Hx, float *Hy, float *Hz) {
	int tk, tj;
	int i, j, k;
	tk = threadIdx.x;
	tj = threadIdx.y;
	k = TPBx*blockIdx.x + tk;
	j = TPBy*(blockIdx.y % BPGy ) + tj;
	i = blockIdx.y / BPGy;

	if ( i<Nx && j<Ny && k<Nz ) {
		int Nyzpit = Ny*Nzpit;
		int idx = k + Nzpit*j + Nyzpit*k;

		__shared__ float ex[TPBy+1][TPBx+1], ey[TPBy][TPBx+1], ez[TPBy+1][TPBx];
		ex[tj+1][tk+1] = Ex[idx];
		ey[tj][tk+1] = Ey[idx];
		ez[tj+1][tk] = Ez[idx];
		if ( tk==0 && k>0 ) {
			ex[tj][0] = Ex[idx-1];
			ey[tj][0] = Ey[idx-1];
		}
		if ( tj==0 && j>0 ) {
			ex[0][tk] = Ex[idx-Nzpit];
			ez[0][tk] = Ez[idx-Nzpit];
		}
		__syncthreads();

		if ( k < Nz ) {
			if ( j>0 && k>0 ) Hx[idx] -= 0.5*( ez[tj+1][tk] - ez[tj][tk] - ey[tj][tk+1] + ey[tj][tk] );
			if ( i>0 && k>0 ) Hy[idx] -= 0.5*( ex[tj+1][tk+1] - ex[tj+1][tk] - ez[tj+1][tk] + Ez[idx-Nyzpit] );
			if ( i>0 && j>0 ) Hz[idx] -= 0.5*( ey[tj][tk+1] - Ey[idx-Nyzpit] - ex[tj+1][tk+1] + ex[tj][tk+1] );
		}
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
	//Ny = 16;
	//Nz = 20;
	TMAX = 100;

	// Allocate host memory
	float ***Ex;
	float ***CEx, ***CEy, ***CEz;
	Ex = makeArray(Nx, Ny, Nz);
	CEx = makeArray(Nx, Ny, Nz);
	CEy = makeArray(Nx, Ny, Nz);
	CEz = makeArray(Nx, Ny, Nz);

	// Geometry
	set_geometry(Nx, Ny, Nz, CEx, CEy, CEz);


	// Allocate device memory
	float *devEx, *devEy, *devEz;
	float *devHx, *devHy, *devHz;
	float *devCEx, *devCEy, *devCEz;
	int z_size = Nz*sizeof(float);
	size_t pitch;
	cudaMallocPitch ( (void**) &devEx, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devEy, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devEz, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devCEx, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devCEy, &pitch, z_size, Nx*Ny );
	CUDAmallocPitch ( (void**) &devCEz, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devHx, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devHy, &pitch, z_size, Nx*Ny );
	cudaMallocPitch ( (void**) &devHz, &pitch, z_size, Nx*Ny );
	
	// Copy arrays from host to device
	cudaMemcpy2D ( devCEx, pitch, CEx[0][0], z_size, z_size, Nx*Ny, cudaMemcpyHostToDevice );
	cudaMemcpy2D ( devCEy, pitch, CEy[0][0], z_size, z_size, Nx*Ny, cudaMemcpyHostToDevice );
	cudaMemcpy2D ( devCEz, pitch, CEz[0][0], z_size, z_size, Nx*Ny, cudaMemcpyHostToDevice );
	
	int Nz_pitch = pitch/4;
	printf("pitch= %u, Nz_pitch= %d\n", pitch, Nz_pitch);

	// Number of thread blocks in the grid
	// Number of threads per block
	int BPGx = Nz_pitch/TPBx;
	int BPGy = Ny/TPBy == 0 ? Ny/TPBy : Ny/TPBy + 1;
	int BPGz = Nx;
	//dim3 Dg(BPGx, BPGy*BPGz);
	dim3 Dg = dim3(BPGx, BPGy*BPGz);
	dim3 Db = dim3(TPBx, TPBy);
	//printf("TPBx=%d, TPBy=%d, TPBz=%d\n", TPBx, TPBy, TPBz);
	printf("TPBx=%d, TPBy=%d\n", TPBx, TPBy);
	printf("BPGx=%d, BPGy=%d, BPGz=%d\n", BPGx, BPGy, BPGz);
	printf("Dg(%d,%d)\n", BPGx, BPGy*BPGz);
	printf("Db(%d,%d)\n", TPBx, TPBy);
	printf("Treads per block: %d (%d,%d,%d)\n", TPBx*TPBy*1, TPBx, TPBy, 1);
	if ( TPBx*TPBy > 512 ) {
		printf("Error: An excessive number of threads per block.\n%d (%d,%d,%d)\n", TPBx*TPBy, TPBx, TPBy, 1);
		exit(0);
	}
	printf("Blocks per grid: %d (%d,%d,%d)\n", BPGx*BPGy*BPGz, BPGx, BPGy, BPGz);
	if ( BPGx*BPGy*BPGz > 65535 ) {
		printf("Error: An excessive number of blocks per grid.\n%d (%d,%d,%d)\n", BPGx*BPGy*BPGz, BPGx, BPGy, BPGz);
		exit(0);
	}

	int TPBsrc = Nx;
	int BPGsrc = 1;
	dim3 Dgsrc(BPGsrc);
	dim3 Dbsrc(TPBsrc);

	int N = Nx*Ny*Nz_pitch;
	int TPBinit = Nz_pitch;
	int BPGinit = N%TPBinit == 0 ? N/TPBinit : N/TPBinit + 1;
	dim3 Dginit(BPGinit);
	dim3 Dbinit(TPBinit);

	// Initialize the device arrays
	//initArrays <<<Dginit,Dbinit>>> ( Nx, Ny, Nz_pitch, devEx, devEy, devEz, devHx, devHy, devHz );

	// Main time loop
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++) {
	//for ( tstep=1; tstep<=10; tstep++) {
		// Update on the GPU
		updateE <<<Dg,Db>>> ( Nx, Ny, Nz, Nz_pitch, BPGy, devEx, devEy, devEz, devHx, devHy, devHz, devCEx, devCEy, devCEz );
		//updateSrc <<<Dgsrc,Dbsrc>>> ( Nx, Ny, Nz, Nz_pitch, devEx, tstep );
		//updateH <<<Dg,Db>>> ( Nx, Ny, Nz, Nz_pitch, BPGy, devEx, devEy, devEz, devHx, devHy, devHz );

		
		//if ( tstep/10*10 == tstep ) {
			// Copy arrays from device to host
			cudaMemcpy2D( Ex[0][0], z_size, devEx, pitch, z_size, Nx*Ny, cudaMemcpyDeviceToHost );

			//print_array(Nx, Ny, Nz, Ex);
			//dumpToH5(Nx, Ny, Nz, Nx/2, 0, 0, Nx/2, Ny-1, Nz-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		//}
		
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
}
