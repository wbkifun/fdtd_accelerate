#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

#define Npml 10


const float light_velocity = 2.99792458e8;	// m s- 
const float ep0 = 8.85418781762038920e-12;	// F m-1 (permittivity at vacuum)
const float	mu0 = 1.25663706143591730e-6;	// N A-2 (permeability at vacuum)
const float imp0 = sqrt( mu0/ep0 );	// (impedance at vacuum)
const float pi = 3.14159265358979323846;


// Allocate constant memory for CPML
__constant__ float rcmbEf[Npml];
__constant__ float rcmaEf[Npml];
__constant__ float rcmbEb[Npml];
__constant__ float rcmaEb[Npml];
__constant__ float rcmbHf[Npml];
__constant__ float rcmaHf[Npml];
__constant__ float rcmbHb[Npml];
__constant__ float rcmaHb[Npml];


typedef struct N3 {
	int x, y, z;
} N3;


typedef struct P3F3 {
	float ***x, ***y, ***z;
} P3F3;


typedef struct P1F3 {
	float *x, *y, *z;
} P1F3;


typedef struct P1F2 {
	float *f, *b;
} P1F2;


typedef struct P1F6 {
	P1F2 x, y, z;
} P1F6;


__host__ void verify_over_TPB(int TPB) {
	if ( TPB > 512 ) {
		printf("Error: An excessive number of threads per block.\n");
		exit(0);
	}
}


__host__ void verify_over_BPG(int BPG) {
	if ( BPG > 65535 ) {
		printf("Error: An excessive number of blocks per grid.\n");
		exit(0);
	}
}


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


__host__ void set_geometry(N3 N, P3F3 C) {
	int i,j,k;

	for (i=0; i<N.x; i++) {
		for (j=0; j<N.y; j++) {
			for (k=0; k<N.z; k++) {
				C.x[i][j][k] = 0.5;
				C.y[i][j][k] = 0.5;
				C.z[i][j][k] = 0.5;
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


__global__ void initArray(int Ntot, float *a) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if ( idx < Ntot ) a[idx] = 0;
}


__host__ void initMainArrays(N3 N, int Nzpit, P1F3 F) {
	int TPB=512;
	int Ntot, BPG;
	Ntot = N.x*N.y*Nzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	verify_over_BPG( BPG );

	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, F.x); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, F.y); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, F.z); 
}


__host__ void initPsiArrays(N3 N, int Nzpit, P1F6 psix, P1F6 psiy, P1F6 psiz) {
	int TPB=512;
	int Ntot, BPG;
	Ntot = Npml*N.y*Nzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	verify_over_BPG( BPG );
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psix.y.f); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psix.y.b); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psix.z.f); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psix.z.b); 
	
	Ntot = N.x*Npml*Nzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	verify_over_BPG( BPG );
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiy.z.f); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiy.z.b); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiy.x.f); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiy.x.b); 

	Ntot = N.x*N.y*Npml;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	verify_over_BPG( BPG );
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiz.x.f); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiz.x.b); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiz.y.f); 
	initArray <<<dim3(BPG),dim3(TPB)>>> (Ntot, psiz.y.b); 
}


__global__ void updateE(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE) {
	int tk, idx;
	tk = threadIdx.x;
	idx = blockIdx.x*TPB + tk;

	int i,j,k;
	int Nyz = N.y*Nzpit;
	i = idx/Nyz;
	j = ( idx - i*Nyz )/Nzpit;
	k = idx - i*Nyz - j*Nzpit;

	//printf("[%.2d]\t\t[%.2d]\t\t[%.2d,%.2d,%.2d]\t\t[%.2d]\n", blockIdx.x, tk, i, j, k, idx);

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[TPB+1];
	float* hz = (float*) &hy[TPB+1];

	if ( i<N.x && j<N.y && k<N.z) {
		hx[tk] = H.x[idx];
		hy[tk] = H.y[idx];
		hz[tk] = H.z[idx];
		if ( tk==TPB-1 && k<N.z-1 ) {
			hx[tk+1] = H.x[idx+1];
			hy[tk+1] = H.y[idx+1];
		}
	}
	__syncthreads();

	if ( i<N.x && j<N.y && k<N.z) {
		if ( j<N.y-1 && k<N.z-1 ) E.x[idx] += CE.x[idx]*( H.z[idx+Nzpit] - hz[tk] - hy[tk+1] + hy[tk] );
		if ( i<N.x-1 && k<N.z-1 ) E.y[idx] += CE.y[idx]*( hx[tk+1] - hx[tk] - H.z[idx+Nyz] + hz[tk] );
		if ( i<N.x-1 && j<N.y-1 ) E.z[idx] += CE.z[idx]*( H.y[idx+Nyz] - hy[tk] - H.x[idx+Nzpit] + hx[tk] );
	}
}


__global__ void updateSrc(N3 N, int Nzpit, P1F3 E, int tstep) {
	int idx, ijk;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	//ijk = idx*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + (N.z/2);
	ijk = (N.x/2)*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + idx;

	//if ( idx < N.x ) E.x[ijk] += sin(0.1*tstep);
	if ( idx < N.z ) E.z[ijk] += sin(0.1*tstep);
}


__global__ void updateH(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H) {
	int tk, idx;
	tk = threadIdx.x;
	idx = blockIdx.x*TPB + tk;

	int i,j,k;
	int Nyz = N.y*Nzpit;
	i = idx/Nyz;
	j = ( idx - i*Nyz )/Nzpit;
	k = idx - i*Nyz - j*Nzpit;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[TPB+1];
	float* ez = (float*) &ey[TPB+1];

	if ( i<N.x && j<N.y && k<N.z) {
		ex[tk+1] = E.x[idx];
		ey[tk+1] = E.y[idx];
		ez[tk] = E.z[idx];
		if ( tk==0 && k>0 ) {
			ex[0] = E.x[idx-1];
			ey[0] = E.y[idx-1];
		}
	}
	__syncthreads();

	if ( i<N.x && j<N.y && k<N.z) {
		if ( j>0 && k>0 ) H.x[idx] -= 0.5*( ez[tk] - E.z[idx-Nzpit] - ey[tk+1] + ey[tk] );
		if ( i>0 && k>0 ) H.y[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + E.z[idx-Nyz] );
		if ( i>0 && j>0 ) H.z[idx] -= 0.5*( ey[tk+1] - E.y[idx-Nyz] - ex[tk+1] + E.x[idx-Nzpit] );
	}
}


__global__ void updateCPMLxEf(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int idx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = N.y*Nzpit;
	int i = idx/Nyz;

	if ( i<Npml ) {
		psi.y.f[idx] = rcmbEf[i]*psi.y.f[idx] + rcmaEf[i]*( H.z[idx+Nyz] - H.z[idx] );
		E.y[idx] -= CE.y[idx]*psi.y.f[idx];

		psi.z.f[idx] = rcmbEf[i]*psi.z.f[idx] + rcmaEf[i]*( H.y[idx+Nyz] - H.y[idx] );
		E.z[idx] += CE.z[idx]*psi.z.f[idx];
	}
}


__global__ void updateCPMLxEb(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = N.y*Nzpit;
	int i = pidx/Nyz;

	int idx0 = Nyz*(N.x-Npml-1);
	int idx = pidx + idx0;

	//printf("[%d]\n", i);
	if ( i<N.x-1 ) {
		//printf("[%.5d]\t[%.5d]\t[%d]\trcmbEb=%g\trcmaEb=%g\n", pidx, idx, i, rcmbEb[i], rcmaEb[i]);
		psi.y.b[pidx] = rcmbEb[i]*psi.y.b[pidx] + rcmaEb[i]*( H.z[idx+Nyz] - H.z[idx] );
		E.y[idx] -= CE.y[idx]*psi.y.b[pidx];

		psi.z.b[pidx] = rcmbEb[i]*psi.z.b[pidx] + rcmaEb[i]*( H.y[idx+Nyz] - H.y[idx] );
		E.z[idx] += CE.z[idx]*psi.z.b[pidx];
	}
}


__global__ void updateCPMLxHf(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int idx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = N.y*Nzpit;
	int i = idx/Nyz;

	if ( i<Npml && i>0 ) {
		//printf("[%.5d]\t[%d]\trcmbHf=%g\trcmaHf=%g\n", idx, i, rcmbHf[i], rcmaHf[i]);
		psi.y.f[idx] = rcmbHf[i]*psi.y.f[idx] + rcmaHf[i]*( E.z[idx] - E.z[idx-Nyz] );
		H.y[idx] += 0.5*psi.y.f[idx];

		psi.z.f[idx] = rcmbHf[i]*psi.z.f[idx] + rcmaHf[i]*( E.y[idx] - E.y[idx-Nyz] );
		H.z[idx] -= 0.5*psi.z.f[idx];
	}
}


__global__ void updateCPMLxHb(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = N.y*Nzpit;
	int i = pidx/Nyz;

	int idx0 = Nyz*(N.x-Npml);
	int idx = pidx + idx0;

	if ( i<N.x ) {
		//printf("[%.5d]\t[%.5d]\t[%d]\trcmbHb=%g\trcmaHb=%g\n", pidx, idx, i, rcmbHb[i], rcmaHb[i]);
		psi.y.b[pidx] = rcmbHb[i]*psi.y.b[pidx] + rcmaHb[i]*( E.z[idx] - E.z[idx-Nyz] );
		H.y[idx] += 0.5*psi.y.b[pidx];

		psi.z.b[pidx] = rcmbHb[i]*psi.z.b[pidx] + rcmaHb[i]*( E.y[idx] - E.y[idx-Nyz] );
		H.z[idx] -= 0.5*psi.z.b[pidx];
	}
}


__global__ void updateCPMLyEf(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = Npml*Nzpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpit;

	int idx0 = Nzpit*(N.y-Npml);
	int idx = pidx + i*idx0;
	//printf("[%.5d]\t[%d]\trcmbEf=%g\trcmaEf=%g\n", idx, j, rcmbEf[j], rcmaEf[j]);
	psi.z.f[pidx] = rcmbEf[j]*psi.z.f[pidx] + rcmaEf[j]*( H.x[idx+Nzpit] - H.x[idx] );
	E.z[idx] -= CE.z[idx]*psi.z.f[pidx];

	psi.x.f[pidx] = rcmbEf[j]*psi.x.f[pidx] + rcmaEf[j]*( H.z[idx+Nzpit] - H.z[idx] );
	E.x[idx] += CE.x[idx]*psi.x.f[pidx];
}


__global__ void updateCPMLyEb(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = Npml*Nzpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpit;
	
	//int idx0 = Nzpit*(N.y-1-Npml);
	//int idx = pidx + (i+1)*idx0 + i*Nzpit;
	int idx0 = Nzpit*(N.y-Npml);
	int idx = pidx + (i+1)*idx0 - Nzpit;

	//int jt = ( idx - i*(N.y*Nzpit) )/Nzpit;
	//printf("[%.5d]\t[%d]\t[%d]\trcmbEb=%g\trcmaEb=%g\n", idx, jt, j, rcmbEb[j], rcmaEb[j]);
	
	psi.z.b[pidx] = rcmbEb[j]*psi.z.b[pidx] + rcmaEb[j]*( H.x[idx+Nzpit] - H.x[idx] );
	E.z[idx] -= CE.z[idx]*psi.z.b[pidx];
	
	psi.x.b[pidx] = rcmbEb[j]*psi.x.b[pidx] + rcmaEb[j]*( H.z[idx+Nzpit] - H.z[idx] );
	E.x[idx] += CE.x[idx]*psi.x.b[pidx];
}


__global__ void updateCPMLyHf(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = Npml*Nzpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpit;

	int idx0 = Nzpit*(N.y-Npml);
	int idx = pidx + i*idx0;
	
	if ( j>0 ) {
		//printf("[%.5d][%.5d]\t[%d]\trcmbHf=%g\trcmaHf=%g\n", pidx, idx, j, rcmbHf[j], rcmaHf[j]);
		psi.z.f[pidx] = rcmbHf[j]*psi.z.f[pidx] + rcmaHf[j]*( E.x[idx] - E.x[idx-Nzpit] );
		H.z[idx] += 0.5*psi.z.f[pidx];

		psi.x.f[pidx] = rcmbHf[j]*psi.x.f[pidx] + rcmaHf[j]*( E.z[idx] - E.z[idx-Nzpit] );
		H.x[idx] -= 0.5*psi.x.f[pidx];
	}
}


__global__ void updateCPMLyHb(N3 N, int Nzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int Nyz = Npml*Nzpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpit;
	
	int idx0 = Nzpit*(N.y-Npml);
	int idx = pidx + (i+1)*idx0;
	//int idx0 = Nzpit*(N.y-Npml);
	//int idx = pidx + i*idx0 + idx0-Nzpit;

	//int jt = ( idx - i*(N.y*Nzpit) )/Nzpit;
	//printf("[%.5d]\t[%d]\t[%d]\trcmbEb=%g\trcmaEb=%g\n", idx, jt, j, rcmbEb[j], rcmaEb[j]);
	
	psi.z.b[pidx] = rcmbHb[j]*psi.z.b[pidx] + rcmaHb[j]*( E.x[idx] - E.x[idx-Nzpit] );
	H.z[idx] += 0.5*psi.z.b[pidx];

	psi.x.b[pidx] = rcmbHb[j]*psi.x.b[pidx] + rcmaHb[j]*( E.z[idx] - E.z[idx-Nzpit] );
	H.x[idx] -= 0.5*psi.x.b[pidx];
}


__global__ void updateCPMLzEf(N3 N, int Nzpit, int Nzpmlpit, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*Nzpmlpit + threadIdx.x;
	int Nyz = N.y*Nzpmlpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpmlpit;
	int k = pidx - i*Nyz - j*Nzpmlpit;

	int idx0 = Nzpit-Nzpmlpit;
	int idx = pidx + (j+i*N.y)*idx0;

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[Npml+1];

	if ( k<Npml+1 ) {
		hx[k] = H.x[idx];
		hy[k] = H.y[idx];
	}
	__syncthreads();

	//printf("k=%d\n", k);
	if ( i<N.x && j<N.y && k<Npml ) {
	//if ( k<Npml ) {
		//printf("[%.5d][%.5d]\t[%d]\t[%d]\trcmbEf=%g\trcmaEf=%g\n", pidx, idx, j, k, rcmbEf[k], rcmaEf[k]);
		psi.x.f[pidx] = rcmbEf[k]*psi.x.f[pidx] + rcmaEf[k]*( hy[k+1] - hy[k] );
		E.x[idx] -= CE.x[idx]*psi.x.f[pidx];

		psi.y.f[pidx] = rcmbEf[k]*psi.y.f[pidx] + rcmaEf[k]*( hx[k+1] - hx[k] );
		E.y[idx] += CE.y[idx]*psi.y.f[pidx];
	}
}


__global__ void updateCPMLzEb(N3 N, int Nzpit, int Nzpmlpit, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*Nzpmlpit + threadIdx.x;
	int Nyz = N.y*Nzpmlpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpmlpit;
	int k = pidx - i*Nyz - j*Nzpmlpit;

	int idx0 = N.z - Npml - 1;
	int idx = pidx + idx0 + j*(Nzpit-Nzpmlpit-1);

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[Npml+1];

	if ( k<Npml+1 ) {
		hx[k] = H.x[idx];
		hy[k] = H.y[idx];
	}
	__syncthreads();

	//printf("k=%d\n", k);
	if ( k<Npml ) {
		//printf("[%.5d]\t[%d]\t[%d]\trcmbEb=%g\trcmaEb=%g\n", idx, j, k, rcmbEb[k], rcmaEb[k]);
		psi.x.b[pidx] = rcmbEb[k]*psi.x.b[pidx] + rcmaEb[k]*( hy[k+1] - hy[k] );
		E.x[idx] -= CE.x[idx]*psi.x.b[pidx];

		psi.y.b[pidx] = rcmbEb[k]*psi.y.b[pidx] + rcmaEb[k]*( hx[k+1] - hx[k] );
		E.y[idx] += CE.y[idx]*psi.y.b[pidx];
	}
}


__global__ void updateCPMLzHf(N3 N, int Nzpit, int Nzpmlpit, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*Nzpmlpit + threadIdx.x;
	int Nyz = N.y*Nzpmlpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpmlpit;
	int k = pidx - i*Nyz - j*Nzpmlpit;

	int idx0 = Nzpit-Nzpmlpit;
	int idx = pidx + (j+i*N.y)*idx0;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[Npml+1];

	if ( k<Npml+1 ) {
		ex[k+1] = E.x[idx];
		ey[k+1] = E.y[idx];
	}
	__syncthreads();

	//printf("k=%d\n", k);
	//if ( k<Npml && k>0 ) {
	if ( i<N.x && j<N.y && k<Npml ) {
		//printf("[%.5d]\t[%d]\trcmbHf=%g\trcmaHf=%g\n", idx, k, rcmbHf[k], rcmaHf[k]);
		psi.x.f[pidx] = rcmbHf[k]*psi.x.f[pidx] + rcmaHf[k]*( ey[k+1] - ey[k] );
		H.x[idx] += 0.5*psi.x.f[pidx];

		psi.y.f[pidx] = rcmbHf[k]*psi.y.f[pidx] + rcmaHf[k]*( ex[k+1] - ex[k] );
		H.y[idx] -= 0.5*psi.y.f[pidx];
	}
}


__global__ void updateCPMLzHb(N3 N, int Nzpit, int Nzpmlpit, P1F3 E, P1F3 H, P1F3 CE, P1F6 psi) {
	int pidx = blockIdx.x*Nzpmlpit + threadIdx.x;
	int Nyz = N.y*Nzpmlpit;
	int i = pidx/Nyz;
	int j = ( pidx - i*Nyz )/Nzpmlpit;
	int k = pidx - i*Nyz - j*Nzpmlpit;

	int idx0 = N.z - Npml;
	int idx = pidx + idx0 + j*(Nzpit-Nzpmlpit);

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[Npml+1];

	if ( k<Npml+1 ) {
		ex[k+1] = E.x[idx];
		ey[k+1] = E.y[idx];
	}
	__syncthreads();

	//printf("k=%d\n", k);
	if ( k<Npml ) {
		//printf("[%.5d]\t[%d]\t[%d]\trcmbHb=%g\trcmaHb=%g\n", idx, j, k, rcmbHb[k], rcmaHb[k]);
		psi.x.b[pidx] = rcmbHb[k]*psi.x.b[pidx] + rcmaHb[k]*( ey[k+1] - ey[k] );
		H.x[idx] += 0.5*psi.x.b[pidx];

		psi.y.b[pidx] = rcmbHb[k]*psi.y.b[pidx] + rcmaHb[k]*( ex[k+1] - ex[k] );
		H.y[idx] -= 0.5*psi.y.b[pidx];
	}
}



int main() {
	int tstep;
	char time_str[32];
	time_t t0;
	int i;

	// Set the parameters
	N3 N;
	N.x = 100;
	N.y = 150;
	N.z = 250;
	//N.y = 30;
	//N.z = 20;
	int TMAX = 1000;

	float S = 0.5;
	float dx = 10e-9;
	float dt = S*dx/light_velocity;
	printf("NPML=%d\n", Npml);
	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);

	// Allocate host memory
	P3F3 CE;
	CE.x = makeArray(N);
	CE.y = makeArray(N);
	CE.z = makeArray(N);
	float ***Ex, ***Ez;
	Ex = makeArray(N);
	Ez = makeArray(N);

	// Geometry
	set_geometry(N, CE);

	// CPML
	int m = 4;	// grade_order
	float sigma_max = (m+1.)/(15*pi*Npml*dx);
	float alpha = 0.05;
	P1F2 sigmaE, bE, aE;
	P1F2 sigmaH, bH, aH;

	sigmaE.f = (float *) calloc (Npml, sizeof(float));
	sigmaE.b = (float *) calloc (Npml, sizeof(float));
	sigmaH.f = (float *) calloc (Npml, sizeof(float));
	sigmaH.b = (float *) calloc (Npml, sizeof(float));
	bE.f = (float *) calloc (Npml, sizeof(float));
	bE.b = (float *) calloc (Npml, sizeof(float));
	bH.f = (float *) calloc (Npml, sizeof(float));
	bH.b = (float *) calloc (Npml, sizeof(float));
	aE.f = (float *) calloc (Npml, sizeof(float));
	aE.b = (float *) calloc (Npml, sizeof(float));
	aH.f = (float *) calloc (Npml, sizeof(float));
	aH.b = (float *) calloc (Npml, sizeof(float));
	for (i=0; i<Npml; i++) {
		sigmaE.f[i] = pow( (Npml-0.5-i)/Npml, m )*sigma_max;
		sigmaE.b[i] = pow( (0.5+i)/Npml, m )*sigma_max;
		sigmaH.f[i] = pow( (float)(Npml-i)/Npml, m )*sigma_max;
		sigmaH.b[i] = pow( (1.+i)/Npml, m )*sigma_max;

		bE.f[i] = exp( -(sigmaE.f[i] + alpha)*dt/ep0 );
		bE.b[i] = exp( -(sigmaE.b[i] + alpha)*dt/ep0 );
		bH.f[i] = exp( -(sigmaH.f[i] + alpha)*dt/ep0 );
		bH.b[i] = exp( -(sigmaH.b[i] + alpha)*dt/ep0 );
		aE.f[i] = sigmaE.f[i]/(sigmaE.f[i]+alpha)*(bE.f[i]-1);
		aE.b[i] = sigmaE.b[i]/(sigmaE.b[i]+alpha)*(bE.b[i]-1);
		aH.f[i] = sigmaH.f[i]/(sigmaH.f[i]+alpha)*(bH.f[i]-1);
		aH.b[i] = sigmaH.b[i]/(sigmaH.b[i]+alpha)*(bH.b[i]-1);
		//printf("[%d]\tsigmaE.f=%g,\tbE.f=%g,aE.f=%g\n", i, sigmaE.f[i], bE.f[i], aE.f[i]);
		printf("[%d]\tsigmaE.b=%g,\tbE.b=%g,aE.b=%g\n", i, sigmaE.b[i], bE.b[i], aE.b[i]);
		//printf("[%d]\tsigmaH.f=%g,\tbH.f=%g,aH.f=%g\n", i, sigmaH.f[i], bH.f[i], aH.f[i]);
		//printf("[%d]\tsigmaH.b=%g,\tbH.b=%g,aH.b=%g\n", i, sigmaH.b[i], bH.b[i], aH.b[i]);
	}
	free(sigmaE.f);
	free(sigmaE.b);
	free(sigmaH.f);
	free(sigmaH.b);

	// Copy arrays from host to constant memory
	cudaMemcpyToSymbol(rcmbEf, bE.f, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmaEf, aE.f, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmbEb, bE.b, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmaEb, aE.b, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmbHf, bH.f, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmaHf, aH.f, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmbHb, bH.b, Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmaHb, aH.b, Npml*sizeof(float));

	// Allocate device memory
	P1F3 devE, devH;
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
	
	// Allocate device memory for CPML
	P1F6 psixE, psiyE, psizE;
	P1F6 psixH, psiyH, psizH;

	cudaMallocPitch ( (void**) &psixE.y.f, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixE.y.b, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixE.z.f, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixE.z.b, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psiyE.z.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyE.z.b, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyE.x.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyE.x.b, &pitch, z_size, N.x*Npml );

	cudaMallocPitch ( (void**) &psixH.y.f, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixH.y.b, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixH.z.f, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixH.z.b, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psiyH.z.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.z.b, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.x.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.x.b, &pitch, z_size, N.x*Npml );

	int z_size_pml = Npml*sizeof(float);
	size_t pitch_pmlz;
	cudaMallocPitch ( (void**) &psizE.x.f, &pitch_pmlz, z_size_pml, N.x*N.y );
	cudaMallocPitch ( (void**) &psizE.x.b, &pitch_pmlz, z_size_pml, N.x*N.y );
	cudaMallocPitch ( (void**) &psizE.y.f, &pitch_pmlz, z_size_pml, N.x*N.y );
	cudaMallocPitch ( (void**) &psizE.y.b, &pitch_pmlz, z_size_pml, N.x*N.y );

	cudaMallocPitch ( (void**) &psizH.x.f, &pitch_pmlz, z_size_pml, N.x*N.y );
	cudaMallocPitch ( (void**) &psizH.x.b, &pitch_pmlz, z_size_pml, N.x*N.y );
	cudaMallocPitch ( (void**) &psizH.y.f, &pitch_pmlz, z_size_pml, N.x*N.y );
	cudaMallocPitch ( (void**) &psizH.y.b, &pitch_pmlz, z_size_pml, N.x*N.y );

	// Copy arrays from host to device memory
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
	printf("Blocks per grid: %d\n", BPG);
	verify_over_TPB( TPB );
	verify_over_BPG( BPG );
	printf("Number of bytes in shared memory: %d\n", Ns);

	//int TPBsrc = N.x;
	int TPBsrc = N.z;
	int BPGsrc = 1;
	dim3 Dgsrc(BPGsrc);
	dim3 Dbsrc(TPBsrc);
	verify_over_TPB( TPBsrc );
	verify_over_BPG( BPGsrc );

	int TPBpmlx = 128;
	int Ntotpmlx = Npml*N.y*Nz_pitch;
	int BPGpmlx = Ntotpmlx%TPBpmlx == 0 ? Ntotpmlx/TPBpmlx : Ntotpmlx/TPBpmlx + 1;
	dim3 Dgpmlx(BPGpmlx);
	dim3 Dbpmlx(TPBpmlx);
	verify_over_BPG( BPGpmlx );

	int TPBpmly = 128;
	int Ntotpmly = N.x*Npml*Nz_pitch;
	int BPGpmly = Ntotpmly%TPBpmly == 0 ? Ntotpmly/TPBpmly : Ntotpmly/TPBpmly + 1;
	dim3 Dgpmly(BPGpmly);
	dim3 Dbpmly(TPBpmly);
	verify_over_BPG( BPGpmly );

	int Nzpml_pitch = pitch_pmlz/4;
	int TPBpmlz = Nzpml_pitch;
	int Ntotpmlz = N.x*N.y*Nzpml_pitch;
	int BPGpmlz = Ntotpmlz%TPBpmlz == 0 ? Ntotpmlz/TPBpmlz : Ntotpmlz/TPBpmlz + 1;
	dim3 Dgpmlz(BPGpmlz);
	dim3 Dbpmlz(TPBpmlz);
	verify_over_BPG( BPGpmlz );
	size_t Nspmlz = sizeof(float)*( 2*(Npml+1) );

	printf("Nzpml_pitch: %d\n", Nzpml_pitch);

	// Initialize the device arrays
	initMainArrays ( N, Nz_pitch, devE );
	initMainArrays ( N, Nz_pitch, devH );
	initPsiArrays ( N, Nz_pitch, psixE, psiyE, psizE );
	initPsiArrays ( N, Nz_pitch, psixH, psiyH, psizH );

	// Main time loop
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++) {
	//for ( tstep=1; tstep<=200; tstep++) {
		// Update on the GPU
		updateE <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH, devCE );
		updateCPMLxEf <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, psixE);
		updateCPMLxEb <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, psixE);
		updateCPMLyEf <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, psiyE);
		updateCPMLyEb <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, psiyE);
		//updateCPMLzEf <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, devCE, psizE);
		//updateCPMLzEb <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, devCE, psizE);

		updateSrc <<<Dgsrc,Dbsrc>>> ( N, Nz_pitch, devE, tstep );

		updateH <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH );
		updateCPMLxHf <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, psixH);
		updateCPMLxHb <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, psixH);
		updateCPMLyHf <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, psiyH);
		updateCPMLyHb <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, psiyH);
		//updateCPMLzHf <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, devCE, psizH);
		//updateCPMLzHb <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, devCE, psizH);
		
		if ( tstep/10*10 == tstep ) {
			// Copy arrays from device to host
			//cudaMemcpy2D( Ex[0][0], z_size, devE.x, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );
			cudaMemcpy2D( Ez[0][0], z_size, devE.z, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );

			//print_array(N, Ex);
			//dumpToH5(N.x, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);
			dumpToH5(N.x, N.y, N.z, 0, 0, N.z/2, N.x-1, N.y-1, N.z/2+1, Ez, "gpu_png/Ez-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ez-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
}
