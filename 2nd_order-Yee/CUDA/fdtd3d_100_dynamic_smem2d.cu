#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>


typedef struct N3 {
	int x, y, z;
} N3;


typedef struct P3F3 {
	float ***x, ***y, ***z;
} P3F3;


typedef struct P1F3 {
	float *x, *y, *z;
} P1F3;


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


__host__ void print_array(N3 N, float ***a) {
	int j,k;
	for (j=0; j<N.y; j++) {
		for (k=0; k<N.z; k++) {
			printf("%1.4f\t", a[N.x/2][j][k]);
		}
		printf("\n");
	}
	printf("\n");
}


__host__ float ***makeArray(N3 N) {
	float ***f;

	f = (float ***) calloc (N.x, sizeof(float **));
	f[0] = (float **) calloc (N.y*N.x, sizeof(float *));
	f[0][0] = (float *) calloc (N.z*N.y*N.x, sizeof(float));

	for (int i=0; i<N.x; i++) f[i] = f[0] + i*N.y;
	for (int i=0; i<N.y*N.x; i++) f[0][i] = f[0][0] + i*N.z;

	return f;
}


__host__ void set_geometry(N3 N, P3F3 CE) {
	int i,j,k;

	for (i=0; i<N.x; i++) {
		for (j=0; j<N.y; j++) {
			for (k=0; k<N.z; k++) {
				CE.x[i][j][k] = 0.5;
				CE.y[i][j][k] = 0.5;
				CE.z[i][j][k] = 0.5;
			}
		}
	}
}


__global__ void initArrays(N3 N, int Nzpit, P1F3 E, P1F3 H) {
	int idx;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	//printf("gridDim.x=%d\n",gridDim.x);
	//printf("blockIdx.x=%d, blockDim.x=%d, threadIdx.x=%d\n", blockIdx.x, blockDim.x, threadIdx.x);
	if ( idx < N.x*N.y*Nzpit ) {
		E.x[idx] = 0;
		E.y[idx] = 0;
		E.z[idx] = 0;
		H.x[idx] = 0;
		H.y[idx] = 0;
		H.z[idx] = 0;
	}
}


__global__ void updateE(N3 N, int Nzpit, N3 TPB, P1F3 E, P1F3 H, P1F3 CE) {
	int idx = (threadIdx.x + TPB.x*blockIdx.x) + ( (threadIdx.x + TPB.x*blockIdx.x)/Nzpit )*Nzpit*(TPB.y-1) + threadIdx.y*Nzpit;
	int Nyzpit = N.y*Nzpit;
	int i = idx/Nyzpit;
	int j = ( idx - i*Nyzpit )/Nzpit;
	int k = idx - i*Nyzpit - j*Nzpit;

	int sidx = threadIdx.x + (TPB.x+1)*threadIdx.y;

	//printf("[%.5d][%.5d][%.5d]\t[%.5d][%.5d]\t\t[%.2d,%.2d,%.2d]\n", blockIdx.x, threadIdx.y, threadIdx.x, sidx, idx, i, j, k);

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[(TPB.x+1)*(TPB.y+1)];
	float* hz = (float*) &hy[(TPB.x+1)*(TPB.y+1)];

	if ( i<N.x && j<N.y && k<N.z) {
		hx[sidx] = H.x[idx];
		hy[sidx] = H.y[idx];
		hz[sidx] = H.z[idx];
		if ( sidx%(TPB.x+1)==TPB.x-1 && k<N.z-1 ) {
			hx[sidx+1] = H.x[idx+1];
			hy[sidx+1] = H.y[idx+1];
		}
		if ( sidx/(TPB.x+1)==TPB.y-1 && j<N.y-1 ) {
			hx[sidx+TPB.x+1] = H.x[idx+Nzpit];
			hz[sidx+TPB.x+1] = H.z[idx+Nzpit];
		}
	}
	__syncthreads();

	if ( i<N.x && j<N.y && k<N.z) {
		//if ( j<N.y-1 && k<N.z-1 ) E.x[idx] += CE.x[idx]*( H.z[idx+Nzpit] - hz[sidx] - hy[sidx+1] + hy[sidx] );
		if ( j<N.y-1 && k<N.z-1 ) E.x[idx] += CE.x[idx]*( hz[sidx+TPB.x+1] - hz[sidx] - hy[sidx+1] + hy[sidx] );
		if ( i<N.x-1 && k<N.z-1 ) E.y[idx] += CE.y[idx]*( hx[sidx+1] - hx[sidx] - H.z[idx+Nyzpit] + hz[sidx] );
		//if ( i<N.x-1 && j<N.y-1 ) E.z[idx] += CE.z[idx]*( H.y[idx+Nyzpit] - hy[sidx] - H.x[idx+Nzpit] + hx[sidx] );
		if ( i<N.x-1 && j<N.y-1 ) E.z[idx] += CE.z[idx]*( H.y[idx+Nyzpit] - hy[sidx] - hx[sidx+TPB.x+1] + hx[sidx] );
		//if ( j<N.y-1 && k<N.z-1 ) E.x[idx] += CE.x[idx]*( hz[sidx+TPB.x+1] - hz[sidx] - hy[sidx+1] + hy[sidx] );
		//if ( i<N.x-1 && k<N.z-1 ) E.y[idx] += CE.y[idx]*( hx[sidx+1] - hx[sidx] - H.z[idx+Nyzpit] + hz[sidx] );
		//if ( i<N.x-1 && j<N.y-1 ) E.z[idx] += CE.z[idx]*( H.y[idx+Nyzpit] - hy[sidx] - hx[sidx+TPB.x+1] + hx[sidx] );
	}
}


__global__ void updateSrc(N3 N, int Nzpit, P1F3 E, int tstep) {
	int idx, ijk;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	ijk = idx*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + (N.z/2);

	//printf("idx=%d, ijk=%d\n", idx, ijk);
	//Ex[ijk] += __sinf(0.1*tstep);
	if ( idx < N.x ) {
		E.x[ijk] += sin(0.1*tstep);
	}
}


__global__ void updateH(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H) {
	int tk, idx;
	tk = threadIdx.x;
	idx = blockIdx.x*TPB + tk;

	int i,j,k;
	int Nyzpit = N.y*Nzpit;
	i = idx/Nyzpit;
	j = ( idx - i*Nyzpit )/Nzpit;
	k = idx - i*Nyzpit - j*Nzpit;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[TPB+1];
	float* ez = (float*) &ey[TPB+1];

	if ( i<N.x && k<N.z) {
		ex[tk+1] = E.x[idx];
		ey[tk+1] = E.y[idx];
		ez[tk] = E.z[idx];
		if ( tk==0 && k>0 ) {
			ex[0] = E.x[idx-1];
			ey[0] = E.y[idx-1];
		}
	}
	__syncthreads();

	if ( i<N.x && k<N.z) {
		if ( j>0 && k>0 ) H.x[idx] -= 0.5*( ez[tk] - E.z[idx-Nzpit] - ey[tk+1] + ey[tk] );
		if ( i>0 && k>0 ) H.y[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + E.z[idx-Nyzpit] );
		if ( i>0 && j>0 ) H.z[idx] -= 0.5*( ey[tk+1] - E.y[idx-Nyzpit] - ex[tk+1] + E.x[idx-Nzpit] );
	}
}


int main() {
	int tstep;
	char time_str[32];
	time_t t0;

	// Set the parameters
	N3 N;
	N.x = 200;
	N.y = 200;
	N.z = 200;
	//N.y = 16;
	//N.z = 20;
	int TMAX = 10000;
	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);

	// Allocate host memory
	P3F3 CE;
	CE.x = makeArray(N);
	CE.y = makeArray(N);
	CE.z = makeArray(N);
	float ***Ex;
	Ex = makeArray(N);

	// Geometry
	set_geometry(N, CE);


	// Allocate device memory
	P1F3 devE;
	P1F3 devH;
	P1F3 devCE;
	int z_size = N.z*sizeof(float);
	size_t pitch;
	cudaMallocPitch ( (void**) &devE.x, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devE.y, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devE.z, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devH.x, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devH.y, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devH.z, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devCE.x, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devCE.y, &pitch, z_size, N.x*N.y );
	cudaMallocPitch ( (void**) &devCE.z, &pitch, z_size, N.x*N.y );
	
	// Copy arrays from host to device
	cudaMemcpy2D ( devCE.x, pitch, CE.x[0][0], z_size, z_size, N.x*N.y, cudaMemcpyHostToDevice );
	cudaMemcpy2D ( devCE.y, pitch, CE.y[0][0], z_size, z_size, N.x*N.y, cudaMemcpyHostToDevice );
	cudaMemcpy2D ( devCE.z, pitch, CE.z[0][0], z_size, z_size, N.x*N.y, cudaMemcpyHostToDevice );
	
	int Nz_pitch = pitch/4;
	printf("pitch= %u, Nz_pitch= %d\n", pitch, Nz_pitch);

	// Set the GPU parameters
	int Ntot = N.x*N.y*Nz_pitch;
	int TPB = 256;	// Number of threads per block
	int BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1; // Number of thread blocks per grid
	dim3 Dg = dim3(BPG);
	dim3 Db = dim3(TPB);
	size_t Ns = sizeof(float)*( (TPB+1)+(TPB+1)+(TPB) );
	printf("Threads per block: %d\n", TPB);
	if ( TPB > 512 ) {
		printf("Error: An excessive number of threads per block.\n");
		exit(0);
	}
	printf("Blocks per grid: %d\n", BPG);
	if ( BPG > 65535 ) {
		printf("Error: An excessive number of blocks per grid.\n");
		exit(0);
	}
	printf("Number of bytes in shared memory: %d\n", Ns);

	N3 TPBmain; 
	TPBmain.x = 128;
	TPBmain.y = 4;
	TPBmain.z = 1;
	int BPGmain = Ntot%(TPBmain.x*TPBmain.y) == 0 ? Ntot/(TPBmain.x*TPBmain.y) : Ntot/(TPBmain.x*TPBmain.y) + 1; // Number of thread blocks per grid
	//BPGmain.x = Nz_pitch%TPBmain.x == 0 ? Nz_pitch/TPBmain.x : Nz_pitch/TPBmain.x + 1; // Number of thread blocks per grid
	//BPGmain.y = (N.x*N.y)%TPBmain.y == 0 ? (N.x*N.y)/TPBmain.y : (N.x*N.y)/TPBmain.y + 1; 
	//dim3 Dgmain = dim3(BPGmain.x, BPGmain.y);
	dim3 Dgmain = dim3(BPGmain);
	dim3 Dbmain = dim3(TPBmain.x, TPBmain.y, TPBmain.z);
	//size_t Nsmain = sizeof(float)*( (TPBmain.x+1)*(TPBmain.y+1) + (TPBmain.x+1)*TPBmain.y + TPBmain.x*(TPBmain.y+1) );
	size_t Nsmain = sizeof(float)*( 3*(TPBmain.x+1)*(TPBmain.y+1) );

	int TPBsrc = N.x;
	int BPGsrc = 1;
	dim3 Dgsrc(BPGsrc);
	dim3 Dbsrc(TPBsrc);

	int TPBinit = Nz_pitch;
	int BPGinit = Ntot%TPBinit == 0 ? Ntot/TPBinit : Ntot/TPBinit + 1;
	dim3 Dginit(BPGinit);
	dim3 Dbinit(TPBinit);

	/*
	// Initialize the device arrays
	initArrays <<<Dginit,Dbinit>>> ( N, Nz_pitch, devE, devH );

	// Main time loop
	t0 = time(0);
	//for ( tstep=1; tstep<=TMAX; tstep++) {
	for ( tstep=1; tstep<=200; tstep++) {
		// Update on the GPU
		//updateE <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH, devCE );
		updateE <<<Dgmain,Dbmain,Nsmain>>> ( N, Nz_pitch, TPBmain, devE, devH, devCE );
		updateSrc <<<Dgsrc,Dbsrc>>> ( N, Nz_pitch, devE, tstep );
		updateH <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH );

		if ( tstep/10*10 == tstep ) {
			// Copy arrays from device to host
			cudaMemcpy2D( Ex[0][0], z_size, devE.x, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );

			//print_array(N, Ex);
			dumpToH5(N.x, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
	*/
	for ( tstep=1; tstep<=10; tstep++ ) updateE <<<Dgmain,Dbmain,Nsmain>>> ( N, Nz_pitch, TPBmain, devE, devH, devCE );
	//for ( tstep=1; tstep<=10; tstep++ ) updateH <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH );
}
