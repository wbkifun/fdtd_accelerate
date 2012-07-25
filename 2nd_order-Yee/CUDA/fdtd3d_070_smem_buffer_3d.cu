#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

#define TPBx 16	// Number of threads per block
#define TPBy 4
#define TPBz 4


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


__global__ void updateE(N3 N, int Nzpit, N3 BPG, P1F3 E, P1F3 H, P1F3 CE) {
	int bk;
	int tk, tj, ti;
	int i, j, k;
	bk = blockIdx.x;
	tk = threadIdx.x;
	tj = threadIdx.y;
	ti = threadIdx.z;
	k = TPBx*( bk%BPG.x ) + tk;
	j = TPBy*( (bk/BPG.x)%BPG.y ) + tj;
	i = TPBz*( bk/(BPG.x*BPG.y) ) + ti;

	//printf("bk(%d),\tbk%BPGx (%d),\t(bk/BPGx)%BPGy (%d),\tbk/(BPGx*BPGy) (%d)\n", bk, bk%BPG.x, (bk/BPG.x)%BPG.y, bk/(BPG.x*BPG.y) );
	//printf("blockIdx(%d),\tthreadIdx(%d,%d,%d),\tkji(%d,%d,%d)\n", bk, tk, tj, ti, k, j, i);

	int Nyzpit = N.y*Nzpit;
	int idx = k + Nzpit*j + Nyzpit*i;
	
	__shared__ float hx[TPBz][TPBy+1][TPBx+1];
	__shared__ float hy[TPBz+1][TPBy][TPBx+1];
	__shared__ float hz[TPBz+1][TPBy+1][TPBx];

	if ( i<N.x && j<N.y && k<N.z ) {
		//printf("(%d),\t(%d,%d,%d),\t(%d,%d,%d),\t%d\n", bk, tk, tj, ti, k, j, i, idx);

		hx[ti][tj][tk] = H.x[idx];
		hy[ti][tj][tk] = H.y[idx];
		hz[ti][tj][tk] = H.z[idx];

		if ( tk==TPBx-1 && k<N.z-1 ) {
			hx[ti][tj][tk+1] = H.x[idx+1];
			hy[ti][tj][tk+1] = H.y[idx+1];
		}
		if ( tj==TPBy-1 && j<N.y-1 ) {
			hx[ti][tj+1][tk] = H.x[idx+Nzpit];
			hz[ti][tj+1][tk] = H.z[idx+Nzpit];
		}
		if ( ti==TPBz-1 && i<N.x-1 ) {
			hy[ti+1][tj][tk] = H.y[idx+Nyzpit];
			hz[ti+1][tj][tk] = H.z[idx+Nyzpit];
		}
	}
	__syncthreads();

	if ( i<N.x && j<N.y && k<N.z ) {
		if ( j<N.y-1 && k<N.z-1 ) {
			//if ( j==8 && k==10 ) printf("Ex[%d,8,10]=%g\n", i, E.x[idx]);
				E.x[idx] += CE.x[idx]*( hz[ti][tj+1][tk] - hz[ti][tj][tk] - hy[ti][tj][tk+1] + hy[ti][tj][tk] );
			}
		if ( i<N.x-1 && k<N.z-1 ) E.y[idx] += CE.y[idx]*( hx[ti][tj][tk+1] - hx[ti][tj][tk] - hz[ti+1][tj][tk] + hz[ti][tj][tk] );
		if ( i<N.x-1 && j<N.y-1 ) E.z[idx] += CE.z[idx]*( hy[ti+1][tj][tk] - hy[ti][tj][tk] - hx[ti][tj+1][tk] + hx[ti][tj][tk] );
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


__global__ void updateH(N3 N, int Nzpit, N3 BPG, P1F3 E, P1F3 H) {
	int bk;
	int tk, tj, ti;
	int i, j, k;
	bk = blockIdx.x;
	tk = threadIdx.x;
	tj = threadIdx.y;
	ti = threadIdx.z;
	k = TPBx*( bk%BPG.x ) + tk;
	j = TPBy*( (bk/BPG.x)%BPG.y ) + tj;
	i = TPBz*( bk/(BPG.x*BPG.y) ) + ti;

	int Nyzpit = N.y*Nzpit;
	int idx = k + Nzpit*j + Nyzpit*i;
	
	__shared__ float ex[TPBz][TPBy+1][TPBx+1];
	__shared__ float ey[TPBz+1][TPBy][TPBx+1];
	__shared__ float ez[TPBz+1][TPBy+1][TPBx];

	if ( i<N.x && j<N.y && k<N.z ) {
		ex[ti][tj+1][tk+1] = E.x[idx];
		ey[ti+1][tj][tk+1] = E.y[idx];
		ez[ti+1][tj+1][tk] = E.z[idx];
		if ( tk==0 && k>0 ) {
			ex[ti][tj+1][0] = E.x[idx-1];
			ey[ti+1][tj][0] = E.y[idx-1];
		}
		if ( tj==0 && j>0 ) {
			ex[ti][0][tk+1] = E.x[idx-Nzpit];
			ez[ti+1][0][tk] = E.z[idx-Nzpit];
		}
		if ( ti==0 && i>0 ) {
			ey[0][tj][tk+1] = E.y[idx-Nyzpit];
			ez[0][tj+1][tk] = E.z[idx-Nyzpit];
		}
	}
	__syncthreads();

	if ( i<N.x && j<N.y && k<N.z ) {
		if ( j>0 && k>0 ) H.x[idx] -= 0.5*( ez[ti+1][tj+1][tk] - ez[ti+1][tj][tk] - ey[ti+1][tj][tk+1] + ey[ti+1][tj][tk] );
		if ( i>0 && k>0 ) H.y[idx] -= 0.5*( ex[ti][tj+1][tk+1] - ex[ti][tj+1][tk] - ez[ti+1][tj+1][tk] + ez[ti][tj+1][tk] );
		/*
			if ( j==8 && k==10 ) {
				printf("[%.2d]\t\t[%.2d,%.2d,%.2d]\t\t[%.2d,%.2d,%.2d]\t\t[%d]\n", bk, ti, tj, tk, i, j, k, idx);
				//printf("Ex[%d,%d,%d]=%g\n", i, j, k, E.x[idx]);
				printf("ex[%d,%d,%d]=%g\n", ti, tj+1, tk+1, ex[ti][tj+1][tk+1]);
				printf("ex[%d,%d,%d]=%g\n", ti, tj+1, tk, ex[ti][tj+1][tk]);
				printf("Hy[%d][%d,%d,%d]=%g\n", idx, i, j, k, H.y[idx] );
				printf("\n");
			}
		}
		*/
		if ( i>0 && j>0 ) H.z[idx] -= 0.5*( ey[ti+1][tj][tk+1] - ey[ti][tj][tk+1] - ex[ti][tj+1][tk+1] + ex[ti][tj][tk+1] );
	}
}


int main() {
	int tstep;
	char time_str[32];
	time_t t0;

	// Set the parameters
	N3 N;
	N.x = 100;
	N.y = 200;
	N.z = 500;
	//N.y = 16;
	//N.z = 20;
	int TMAX = 1000;
	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);

	// Allocate host memory
	P3F3 CE;
	CE.x = makeArray(N);
	CE.y = makeArray(N);
	CE.z = makeArray(N);

	/*
	float ***Ex;
	P3F3 H;
	Ex = makeArray(N);
	H.y = makeArray(N);
	H.z = makeArray(N);
	*/

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
	N3 BPG;	// Number of thread blocks per grid
	BPG.x = Nz_pitch/TPBx;
	BPG.y = N.y%TPBy == 0 ? N.y/TPBy : N.y/TPBy + 1;
	BPG.z = N.x%TPBz == 0 ? N.x/TPBz : N.x/TPBz + 1;
	dim3 Dg = dim3(BPG.x*BPG.y*BPG.z);
	dim3 Db = dim3(TPBx, TPBy, TPBz);
	//dim3 Dg = dim3(20);
	//dim3 Db = dim3(16,3,4);
	printf("Threads per block: %d (%d,%d,%d)\n", TPBx*TPBy*TPBz, TPBx, TPBy, TPBz);
	if ( TPBx*TPBy*TPBz > 512 ) {
		printf("Error: An excessive number of threads per block.\n");
		exit(0);
	}
	printf("Blocks per grid: %d (%d,%d,%d)\n", BPG.x*BPG.y*BPG.z, BPG.x, BPG.y, BPG.z);
	if ( BPG.x*BPG.y*BPG.z > 65535 ) {
		printf("Error: An excessive number of blocks per grid.\n");
		exit(0);
	}

	int TPBsrc = N.x;
	int BPGsrc = 1;
	dim3 Dgsrc(BPGsrc);
	dim3 Dbsrc(TPBsrc);

	int Ntot = N.x*N.y*Nz_pitch;
	int TPBinit = Nz_pitch;
	int BPGinit = Ntot%TPBinit == 0 ? Ntot/TPBinit : Ntot/TPBinit + 1;
	dim3 Dginit(BPGinit);
	dim3 Dbinit(TPBinit);

	// Initialize the device arrays
	initArrays <<<Dginit,Dbinit>>> ( N, Nz_pitch, devE, devH );

	// Main time loop
	t0 = time(0);
	//for ( tstep=1; tstep<=TMAX; tstep++) {
	for ( tstep=1; tstep<=10; tstep++) {
		// Update on the GPU
		updateE <<<Dg,Db>>> ( N, Nz_pitch, BPG, devE, devH, devCE );
		//updateSrc <<<Dgsrc,Dbsrc>>> ( N, Nz_pitch, devE, tstep );
		//updateH <<<Dg,Db>>> ( N, Nz_pitch, BPG, devE, devH );

		/*
		//if ( tstep/10*10 == tstep ) {
			// Copy arrays from device to host
			cudaMemcpy2D( Ex[0][0], z_size, devE.x, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );
			//cudaMemcpy2D( H.y[0][0], z_size, devH.y, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );
			//cudaMemcpy2D( H.z[0][0], z_size, devH.z, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );

			//printf("Ex\n");
			//print_array(N, Ex);
			//printf("Hy\n");
			//print_array(N, H.y);
			//printf("Hz\n");
			//print_array(N, H.z);
			dumpToH5(N.x, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		//}
		*/
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
}
