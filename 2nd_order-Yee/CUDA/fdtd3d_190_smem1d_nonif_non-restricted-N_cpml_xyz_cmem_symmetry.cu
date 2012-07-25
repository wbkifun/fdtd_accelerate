#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

#define NPML 10

const float light_velocity = 2.99792458e8;	// m s- 
const float ep0 = 8.85418781762038920e-12;	// F m-1 (permittivity at vacuum)
const float	mu0 = 1.25663706143591730e-6;	// N A-2 (permeability at vacuum)
const float imp0 = sqrt( mu0/ep0 );	// (impedance at vacuum)
const float pi = 3.14159265358979323846;

const int MBPG = 65535;

// Allocate constant memory for CPML
__constant__ float rcmbE[2*NPML+1];
__constant__ float rcmaE[2*NPML+1];
__constant__ float rcmbH[2*NPML+1];
__constant__ float rcmaH[2*NPML+1];


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


__host__ float ***makeArray3D(int Nx, int Ny, int Nz) {
	float ***f;
	int i;

	f = (float ***) calloc (Nx, sizeof(float **));
	f[0] = (float **) calloc (Ny*Nx, sizeof(float *));
	f[0][0] = (float *) calloc (Nz*Ny*Nx, sizeof(float));

	for (i=0; i<Nx; i++) f[i] = f[0] + i*Ny;
	for (i=0; i<Ny*Nx; i++) f[0][i] = f[0][0] + i*Nz;

	return f;
}


__host__ float **makeArray2D(int Nx, int Ny) {
	float **f;

	f = (float **) calloc (Nx, sizeof(float *));
	f[0] = (float *) calloc (Ny*Nx, sizeof(float));

	for (int i=0; i<Nx; i++) f[i] = f[0] + i*Ny;

	return f;
}


__host__ float *makeArray1D( int Nx ) {
	float *f;

	f = (float *) calloc (Nx, sizeof(float));

	return f;
}


__host__ void set_geometry( N3 N, P3F3 CE ) {
	int i,j,k;

	for ( i=1; i<N.x; i++ ) {
		for ( j=1; j<N.y; j++ ) {
			for ( k=1; k<N.z; k++ ) {
				CE.x[i][j][k] = 0.5;
				CE.y[i][j][k] = 0.5;
				CE.z[i][j][k] = 0.5;

				if ( i == N.x-1 ) {
					CE.y[i][j][k] = 0;
					CE.z[i][j][k] = 0;
				}
				if ( j == N.y-1 ) {
					CE.z[i][j][k] = 0;
					CE.x[i][j][k] = 0;
				}
				if ( k == N.z-1 ) {
					CE.x[i][j][k] = 0;
					CE.y[i][j][k] = 0;
				}
			}
		}
	}
	/*
	int idx;
	for ( idx=0; idx<(N.x+1)*N.y*N.z; idx++ ) {
		i = idx/(N.y*N.z);
		j = ( idx - i*N.y*N.z )/N.z;
		k = idx%N.z;
		printf("%d [%d,%d,%d] %g, %g, %g\n", idx, i, j, k, CE.x[0][0][idx], CE.y[0][0][idx], CE.z[0][0][idx] );
	}
	*/
}


__host__ void verify_16xNz(int Nz) {
	int R = Nz%16;
	int N1 = Nz-R; 
	int N2 = N1+16; 
	if ( R == 0 ) printf("Nz is a multiple of 16.\n");
	else {
		printf("Error: Nz is not a multiple of 16.\n");
		printf("Recommend Nz: %d or %d\n", N1, N2);
		exit(0);
	}
}


__host__ float calcOccupancy(int TPB) {
	float occupancy;
	int WPB;	// wrap/block
	int ABPM;	// active block/streaming multiprocessor
	int AWPM;	// active warp/streaming multiprocessor
	int MAX_ABPM = 8;	 
	int MAX_AWPM = 32;	 
	//int MAX_TPM = 1024;	 
	int TPW = 32;	// thread/warp

	WPB = TPB%TPW == 0 ? TPB/TPW : TPB/TPW+1;
	ABPM = MAX_AWPM/WPB < MAX_ABPM ? MAX_AWPM/WPB : MAX_ABPM;
	AWPM = WPB*ABPM;
	occupancy = (float)AWPM/MAX_AWPM;

	return occupancy;
}


__host__ int selectTPB(int Ntot, int Nsurplus_plane) {
	int i;
	int *tpb, bpg, TPB=0;
	int Nsurplus;
	float occupancy, max_occupancy=0;

	int Ntpb = 512/16 + 2;
	tpb = (int *) calloc (Ntpb, sizeof(int));
	tpb[0] = 512;
	tpb[1] = 256;
	tpb[2] = 128;
	for ( i=3; i<Ntpb; i++ ) tpb[i] = tpb[0] - 16*(i-2);
	//for ( i=0; i<Ntpb; i++ ) printf("tpb[%d]=%d\n",i,tpb[i]);
	
	for ( i=0; i<Ntpb; i++) {
		occupancy = calcOccupancy( tpb[i] );
		if ( occupancy > max_occupancy ) {
			max_occupancy = occupancy;
			bpg = Ntot%tpb[i] == 0 ? Ntot/tpb[i] : Ntot/tpb[i] + 1;
			Nsurplus = tpb[i]*bpg - Ntot;
			if ( Nsurplus_plane == 0 )  TPB = tpb[i];
			else if ( Nsurplus <= Nsurplus_plane ) TPB = tpb[i];
		}
	}

	if ( TPB == 0 ) {
		printf("Error: There is not a TPB satisfied the conditions\n");
		exit(0);
	}

	printf("\tNsurplus_plane=%d, Nsurplus=%d\n", Nsurplus_plane, Nsurplus);
	printf("\tNtot=%d, TPB=%d\n", Ntot, TPB);

	return TPB;
}


__global__ void initArray(int Ntot, float *a, int idx0) {
	int idx = idx0 + blockIdx.x*blockDim.x + threadIdx.x;

	if ( idx < Ntot ) a[idx] = 0;
}


__host__ void initMainArrays(int Ntot, P1F3 F) {
	int i;
	int TPB, BPG, NK, sBPG, *idx0;
	dim3 Db, *Dg;

	printf("select TPB,BPG: main init\n");
	TPB = 512;
	Db = dim3( TPB );
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	NK = BPG/MBPG + 1;	// Number of kernel
	sBPG = BPG/NK;
	Dg = (dim3 *) malloc ( NK*sizeof(dim3) );
	idx0 = (int *) malloc ( NK*sizeof(int) );
	for ( i=0; i<NK; i++ ) {
		idx0[i] = TPB*sBPG*i;
		Dg[i] = dim3(sBPG);
	}
	Dg[NK-1] = dim3(sBPG+BPG%NK);
	printf("\tTPB=%d, BPG=%d, sBPG(%d)=%d\n", TPB, BPG, NK, sBPG);

	for ( i=0; i<NK; i++ ) {
		initArray <<<Dg[i],Db>>> (Ntot, F.x, idx0[i]); 
		initArray <<<Dg[i],Db>>> (Ntot, F.y, idx0[i]); 
		initArray <<<Dg[i],Db>>> (Ntot, F.z, idx0[i]); 
	}
}


__host__ void initPsiArrays(int Ntot, dim3 Db, dim3 Dg, P1F2 psi1, P1F2 psi2) {
	initArray <<<Dg,Db>>> (Ntot, psi1.f, 0); 
	initArray <<<Dg,Db>>> (Ntot, psi1.b, 0); 
	initArray <<<Dg,Db>>> (Ntot, psi2.f, 0); 
	initArray <<<Dg,Db>>> (Ntot, psi2.b, 0); 
}


__host__ void freeMainArrays(P1F3 F) {
	cudaFree(F.x);
	cudaFree(F.y);
	cudaFree(F.z);
}


__host__ void freePsiArrays(P1F6 psix, P1F6 psiy, P1F6 psiz) {
	cudaFree(psix.y.f);
	cudaFree(psix.y.b);
	cudaFree(psix.z.f);
	cudaFree(psix.z.b);

	cudaFree(psiy.z.f);
	cudaFree(psiy.z.b);
	cudaFree(psiy.x.f);
	cudaFree(psiy.x.b);

	cudaFree(psiz.x.f);
	cudaFree(psiz.x.b);
	cudaFree(psiz.y.f);
	cudaFree(psiz.y.b);
}


__global__ void updateE(N3 N, P1F3 E, P1F3 H, P1F3 CE, int idx0) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk + idx0;
	int Nyz = N.y*N.z;
	int eidx = idx + Nyz;

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[blockDim.x+1];
	float* hz = (float*) &hy[blockDim.x+1];

	hx[tk] = H.x[idx];
	hy[tk] = H.y[idx];
	hz[tk] = H.z[idx];
	
	if ( tk==blockDim.x-1 ) {
		hx[tk+1] = H.x[idx+1];
		hy[tk+1] = H.y[idx+1];
	}
	__syncthreads();

	E.x[eidx] += CE.x[idx]*( H.z[idx+N.z] - hz[tk] - hy[tk+1] + hy[tk] );
	E.y[eidx] += CE.y[idx]*( hx[tk+1] - hx[tk] - H.z[idx+Nyz] + hz[tk] );
	E.z[eidx] += CE.z[idx]*( H.y[idx+Nyz] - hy[tk] - H.x[idx+N.z] + hx[tk] );
}


__global__ void updateH(N3 N, P1F3 E, P1F3 H, int idx0) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tk + idx0;
	int Nyz = N.y*N.z;
	int eidx = idx + Nyz;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];
	float* ez = (float*) &ey[blockDim.x+1];

	ex[tk+1] = E.x[eidx];
	ey[tk+1] = E.y[eidx];
	ez[tk] = E.z[eidx];
	if ( tk==0 ) {
		ex[0] = E.x[eidx-1];
		ey[0] = E.y[eidx-1];
	}
	__syncthreads();

	H.x[idx] -= 0.5*( ez[tk] - E.z[eidx-N.z] - ey[tk+1] + ey[tk] );
	H.y[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + E.z[eidx-Nyz] );
	H.z[idx] -= 0.5*( ey[tk+1] - E.y[eidx-Nyz] - ex[tk+1] + E.x[eidx-N.z] );
}


__global__ void updateSrc(N3 N, P1F3 E, int tstep) {
	int idx = threadIdx.x;
	int ijk = (idx+1)*N.y*N.z + (N.y/2)*N.z + (N.z/2);
	//int ijk = (idx+1)*N.y*N.z + (N.y/2)*N.z + (N.z/2 - 50);
	//int ijk = (N.x/2+1)*N.y*N.z + (N.y/2)*N.z + idx;

	E.x[ijk] += sin(0.1*tstep);
	//E.z[ijk] += sin(0.1*tstep);
}


__global__ void updateCPMLxE(N3 N, P1F3 E, P1F3 H, P1F3 CE, float *psi1, float *psi2, int backward) {
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;
	int Nyz = N.y*N.z;
	int pi = pidx/Nyz + backward*(NPML+1);

	int idx = pidx + ( 1 + backward*(N.x-NPML-2) )*Nyz;
	int eidx = idx + Nyz;

	psi1[pidx] = rcmbE[pi]*psi1[pidx] + rcmaE[pi]*( H.z[idx+Nyz] - H.z[idx] );
	E.y[eidx] -= CE.y[idx]*psi1[pidx];

	psi2[pidx] = rcmbE[pi]*psi2[pidx] + rcmaE[pi]*( H.y[idx+Nyz] - H.y[idx] );
	E.z[eidx] += CE.z[idx]*psi2[pidx];
}


__global__ void updateCPMLxH(N3 N, P1F3 E, P1F3 H, float *psi1, float *psi2, int backward) {
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;
	int Nyz = N.y*N.z;
	int pi = pidx/Nyz + backward*(NPML+1);

	int idx = pidx + ( 1 + backward*(N.x-NPML-1) )*Nyz;
	int eidx = idx + Nyz;

	psi1[pidx] = rcmbH[pi]*psi1[pidx] + rcmaH[pi]*( E.z[eidx] - E.z[eidx-Nyz] );
	H.y[idx] += 0.5*psi1[pidx];

	psi2[pidx] = rcmbH[pi]*psi2[pidx] + rcmaH[pi]*( E.y[eidx] - E.y[eidx-Nyz] );
	H.z[idx] -= 0.5*psi2[pidx];
}


__global__ void updateCPMLyE(N3 N, P1F3 E, P1F3 H, P1F3 CE, float *psi1, float *psi2, int backward) {
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = pidx/(NPML*N.z);
	int pj = ( pidx/N.z )%NPML + backward*(NPML+1);

	int idx = pidx + ( 1 + i*(N.y-NPML) + backward*(N.y-NPML-2) )*N.z;
	int eidx = idx + N.y*N.z;

	psi1[pidx] = rcmbE[pj]*psi1[pidx] + rcmaE[pj]*( H.x[idx+N.z] - H.x[idx] );
	E.z[eidx] -= CE.z[idx]*psi1[pidx];

	psi2[pidx] = rcmbE[pj]*psi2[pidx] + rcmaE[pj]*( H.z[idx+N.z] - H.z[idx] );
	E.x[eidx] += CE.x[idx]*psi2[pidx];
}


__global__ void updateCPMLyH(N3 N, P1F3 E, P1F3 H, float *psi1, float *psi2, int backward) {
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = pidx/(NPML*N.z);
	int pj = ( pidx/N.z )%NPML + backward*(NPML+1);

	int idx = pidx + ( 1 + i*(N.y-NPML) + backward*(N.y-NPML-1) )*N.z;
	int eidx = idx + N.y*N.z;

	psi1[pidx] = rcmbH[pj]*psi1[pidx] + rcmaH[pj]*( E.x[eidx] - E.x[eidx-N.z] );
	H.z[idx] += 0.5*psi1[pidx];

	psi2[pidx] = rcmbH[pj]*psi2[pidx] + rcmaH[pj]*( E.z[eidx] - E.z[eidx-N.z] );
	H.x[idx] -= 0.5*psi2[pidx];
}


__global__ void updateCPMLzE(N3 N, P1F3 E, P1F3 H, P1F3 CE, float *psi1, float *psi2, int backward) {
	int tk = threadIdx.x;
	int pidx = blockIdx.x*blockDim.x + tk;
	int j = pidx/NPML;
	int pk = pidx%NPML + backward*(NPML+1);

	int idx = pidx + 1 + j*(N.z-NPML) + backward*(N.z-NPML-2);
	int eidx = idx + N.y*N.z;

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[blockDim.x+1];

	hx[tk] = H.x[idx];
	hy[tk] = H.y[idx];
	
	if ( tk == blockDim.x-1 ) {
		hx[tk+1] = H.x[idx+1];
		hy[tk+1] = H.y[idx+1];
	}
	__syncthreads();

	//psi1[pidx] = rcmbE[pk]*psi1[pidx] + rcmaE[pk]*( hy[tk+1] - hy[tk] );
	psi1[pidx] = rcmbE[pk]*psi1[pidx] + rcmaE[pk]*( H.y[idx+1] - H.y[idx] );
	E.x[eidx] -= CE.x[idx]*psi1[pidx];

	//psi2[pidx] = rcmbE[pk]*psi2[pidx] + rcmaE[pk]*( hx[tk+1] - hx[tk] );
	psi2[pidx] = rcmbE[pk]*psi2[pidx] + rcmaE[pk]*( H.x[idx+1] - H.x[idx] );
	E.y[eidx] += CE.y[idx]*psi2[pidx];
	/*
	//printf("tk=%d, pidx=%d, pk=%d, idx=%d, eidx=%d, CEx=%g, Ex=%g, CEy=%g, Ey=%g\n", tk, pidx, pk, idx, eidx, CE.x[idx], E.x[eidx], CE.y[idx], E.y[eidx]);
	int i = idx/(N.y*N.z);
	int j = (idx - i*N.y*N.z)/N.z;
	int k = idx%N.z;
	//printf("[%d,%d,%d] [%d,%d,%d,%d,%d] [%g, %g, %g, %g]\n", i,j,k,tk, pidx, pk, idx, eidx, CE.x[idx], E.x[eidx], CE.y[idx], E.y[eidx]);
	printf("[%d,%d,%d] [%d,%d,%d,%d,%d] [%g, %g]\n", i,j,k,tk, pidx, pk, idx, eidx, CE.x[idx], CE.y[idx]);
	*/
}


__global__ void updateCPMLzH(N3 N, P1F3 E, P1F3 H, float *psi1, float *psi2, int backward) {
	int tk = threadIdx.x;
	int pidx = blockIdx.x*blockDim.x + tk;
	int j = pidx/NPML;
	int pk = pidx%NPML + backward*(NPML+1);

	int idx = pidx + 1 + j*(N.z-NPML) + backward*(N.z-NPML-1);
	int eidx = idx + N.y*N.z;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[blockDim.x+1];

	ex[tk+1] = E.x[eidx];
	ey[tk+1] = E.y[eidx];
	if ( tk == 0 ) {
		ex[0] = E.x[eidx-1];
		ey[0] = E.y[eidx-1];
	}
	__syncthreads();

	//psi1[pidx] = rcmbH[pk]*psi1[pidx] + rcmaH[pk]*( ey[tk+1] - ey[tk] );
	psi1[pidx] = rcmbH[pk]*psi1[pidx] + rcmaH[pk]*( E.y[eidx] - E.y[eidx-1] );
	H.x[idx] += 0.5*psi1[pidx];

	//psi2[pidx] = rcmbH[pk]*psi2[pidx] + rcmaH[pk]*( ex[tk+1] - ex[tk] );
	psi2[pidx] = rcmbH[pk]*psi2[pidx] + rcmaH[pk]*( E.x[eidx] - E.x[eidx-1] );
	H.y[idx] -= 0.5*psi2[pidx];
}



int main() {
	int tstep;
	char time_str[32];
	time_t t0;
	int i;

	// --------------------------------------------------------------------------------
	// Set the parameters
	N3 N;
	N.x = 200;
	N.y = 200;
	N.z = 304;
	int TMAX = 500;
	//N.x = 20;
	//N.y = 20;
	//N.z = 32;
	//int TMAX = 1;
	
	float S = 0.5;
	float dx = 10e-9;
	float dt = S*dx/light_velocity;

	int Npml = NPML;
	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);
	verify_16xNz( N.z );
	printf("Npml=%d\n",Npml);

	// --------------------------------------------------------------------------------
	// Allocate host memory
	P3F3 CE;
	CE.x = makeArray3D( N.x+1, N.y, N.z );
	CE.y = makeArray3D( N.x+1, N.y, N.z );
	CE.z = makeArray3D( N.x+1, N.y, N.z );

	//float **Ex, **Ez;
	//Ex = makeArray2D( N.y, N.z );
	//Ez = makeArray2D( N.x, N.y );
	float ***Ex, ***Ey, ***Ez;
	Ex = makeArray3D( N.x+2, N.y, N.z );
	Ey = makeArray3D( N.x+2, N.y, N.z );
	Ez = makeArray3D( N.x+2, N.y, N.z );
	float ***Hx, ***Hy, ***Hz;
	Hx = makeArray3D( N.x+2, N.y, N.z );
	Hy = makeArray3D( N.x+2, N.y, N.z );
	Hz = makeArray3D( N.x+2, N.y, N.z );

	// --------------------------------------------------------------------------------
	// Geometry
	set_geometry( N, CE );

	// --------------------------------------------------------------------------------
	// Parameters for CPML
	int m = 4;	// grade_order
	float sigma_max = (m+1.)/(15*pi*Npml*dx);
	float alpha = 0.05;
	float *sigmaE, *bE, *aE;
	float *sigmaH, *bH, *aH;

	sigmaE = (float *) calloc (2*Npml+1, sizeof(float));
	sigmaH = (float *) calloc (2*Npml+1, sizeof(float));
	bE = (float *) calloc (2*Npml+1, sizeof(float));
	bH = (float *) calloc (2*Npml+1, sizeof(float));
	aE = (float *) calloc (2*Npml+1, sizeof(float));
	aH = (float *) calloc (2*Npml+1, sizeof(float));
	for (i=0; i<Npml; i++) {
		sigmaE[i] = pow( (Npml-0.5-i)/Npml, m )*sigma_max;
		sigmaE[i+Npml+1] = pow( (0.5+i)/Npml, m )*sigma_max;
		sigmaH[i] = pow( (float)(Npml-i)/Npml, m )*sigma_max;
		sigmaH[i+Npml+1] = pow( (1.+i)/Npml, m )*sigma_max;
	}

	for (i=0; i<2*Npml+1; i++) {
		bE[i] = exp( -(sigmaE[i] + alpha)*dt/ep0 );
		bH[i] = exp( -(sigmaH[i] + alpha)*dt/ep0 );
		aE[i] = sigmaE[i]/(sigmaE[i]+alpha)*(bE[i]-1);
		aH[i] = sigmaH[i]/(sigmaH[i]+alpha)*(bH[i]-1);
		//printf("[%d]\tsigmaE=%g,\tbE=%g,aE=%g\n", i, sigmaE[i], bE[i], aE[i]);
		//printf("[%d]\tsigmaH=%g,\tbH=%g,aH=%g\n", i, sigmaH[i], bH[i], aH[i]);
	}

	free(sigmaE);
	free(sigmaH);

	// --------------------------------------------------------------------------------
	// Copy arrays from host to constant memory
	cudaMemcpyToSymbol(rcmbE, bE, (2*Npml+1)*sizeof(float));
	cudaMemcpyToSymbol(rcmaE, aE, (2*Npml+1)*sizeof(float));
	cudaMemcpyToSymbol(rcmbH, bH, (2*Npml+1)*sizeof(float));
	cudaMemcpyToSymbol(rcmaH, aH, (2*Npml+1)*sizeof(float));

	free(bE);
	free(aE);
	free(bH);
	free(aH);

	// --------------------------------------------------------------------------------
	// Set the GPU parameters
	// TPB: Number of threads per block
	// BPG: Number of thread blocks per grid
	int Ntot, TPB, BPG;
	int NK, sBPG, *idx0;

	// main update
	printf("select TPB,BPG: main\n");
	dim3 Db_main, *Dg_main;
	Ntot = N.x*N.y*N.z;
	TPB = selectTPB( Ntot, N.y*N.z );
	Db_main = dim3( TPB );
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	NK = BPG/MBPG + 1;	// Number of kernel
	sBPG = BPG/NK;
	Dg_main = (dim3 *) malloc ( NK*sizeof(dim3) );
	idx0 = (int *) malloc ( NK*sizeof(int) );
	for ( i=0; i<NK; i++ ) {
		idx0[i] = TPB*sBPG*i;
		Dg_main[i] = dim3(sBPG);
	}
	Dg_main[NK-1] = dim3(sBPG+BPG%NK);
	size_t Ns_main = sizeof(float)*( 2*(TPB+1)+TPB );
	printf("\tBPG=%d, sBPG(%d)=%d, Ns_main=%d\n", BPG, NK, sBPG, Ns_main);
	
	// source 
	TPB = N.x;
	//TPB = N.z;
	BPG = 1;
	dim3 DBsrc(TPB);
	dim3 DGsrc(BPG);
	printf("source: TPB=%d, BPG=%d\n", TPB, BPG);

	// cpml 
	printf("select TPB,BPG: pml x\n");
	dim3 Db_pmlx, Dg_pmlx;
	Ntot = Npml*N.y*N.z;
	TPB = selectTPB( Ntot, N.y*N.z );
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	Db_pmlx = dim3( TPB );
	Dg_pmlx = dim3( BPG );
	int Ntotpmlx = TPB*BPG;
	printf("\tBPG=%d\n", BPG);

	printf("select TPB,BPG: pml y\n");
	dim3 Db_pmly, Dg_pmly;
	Ntot = N.x*Npml*N.z;
	TPB = selectTPB( Ntot, Npml*N.z );
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	Db_pmly = dim3( TPB );
	Dg_pmly = dim3( BPG );
	int Ntotpmly = TPB*BPG;
	printf("\tBPG=%d\n", BPG);

	printf("select TPB,BPG: pml z\n");
	dim3 Db_pmlz, Dg_pmlz;
	//int Npml_pitch = (Npml/16 + 1)*16; 
	Ntot = N.x*N.y*Npml;
	TPB = selectTPB( Ntot, N.y*Npml );
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	Db_pmlz = dim3( TPB );
	Dg_pmlz = dim3( BPG );
	int Ntotpmlz = TPB*BPG;
	size_t Ns_pmlz = sizeof(float)*( 2*(TPB+1) );
	printf("\tBPG=%d, Ns_pmlz=%d\n", BPG, Ns_pmlz);


	// --------------------------------------------------------------------------------
	// Allocate device memory
	P1F3 devE, devH;
	P1F3 devCE;

	int size_devF = (N.x+2)*N.y*N.z*sizeof(float);
	int size_devC = (N.x+1)*N.y*N.z*sizeof(float);

	cudaMalloc ( (void**) &devE.x, size_devF );
	cudaMalloc ( (void**) &devE.y, size_devF );
	cudaMalloc ( (void**) &devE.z, size_devF );
	cudaMalloc ( (void**) &devH.x, size_devF );
	cudaMalloc ( (void**) &devH.y, size_devF );
	cudaMalloc ( (void**) &devH.z, size_devF );
	cudaMalloc ( (void**) &devCE.x, size_devC );
	cudaMalloc ( (void**) &devCE.y, size_devC );
	cudaMalloc ( (void**) &devCE.z, size_devC );
	
	// --------------------------------------------------------------------------------
	// Allocate device memory for CPML
	P1F6 psixE, psiyE, psizE;
	P1F6 psixH, psiyH, psizH;

	int size_psix = Ntotpmlx*sizeof(float);
	int size_psiy = Ntotpmly*sizeof(float);
	int size_psiz = Ntotpmlz*sizeof(float);

	cudaMalloc ( (void**) &psixE.y.f, size_psix );
	cudaMalloc ( (void**) &psixE.y.b, size_psix );
	cudaMalloc ( (void**) &psixE.z.f, size_psix );
	cudaMalloc ( (void**) &psixE.z.b, size_psix );
	cudaMalloc ( (void**) &psixH.y.f, size_psix );
	cudaMalloc ( (void**) &psixH.y.b, size_psix );
	cudaMalloc ( (void**) &psixH.z.f, size_psix );
	cudaMalloc ( (void**) &psixH.z.b, size_psix );

	cudaMalloc ( (void**) &psiyE.z.f, size_psiy );
	cudaMalloc ( (void**) &psiyE.z.b, size_psiy );
	cudaMalloc ( (void**) &psiyE.x.f, size_psiy );
	cudaMalloc ( (void**) &psiyE.x.b, size_psiy );
	cudaMalloc ( (void**) &psiyH.z.f, size_psiy );
	cudaMalloc ( (void**) &psiyH.z.b, size_psiy );
	cudaMalloc ( (void**) &psiyH.x.f, size_psiy );
	cudaMalloc ( (void**) &psiyH.x.b, size_psiy );

	cudaMalloc ( (void**) &psizE.x.f, size_psiz );
	cudaMalloc ( (void**) &psizE.x.b, size_psiz );
	cudaMalloc ( (void**) &psizE.y.f, size_psiz );
	cudaMalloc ( (void**) &psizE.y.b, size_psiz );
	cudaMalloc ( (void**) &psizH.x.f, size_psiz );
	cudaMalloc ( (void**) &psizH.x.b, size_psiz );
	cudaMalloc ( (void**) &psizH.y.f, size_psiz );
	cudaMalloc ( (void**) &psizH.y.b, size_psiz );
	
	// --------------------------------------------------------------------------------
	// Initialize the device arrays
	initMainArrays ( (N.x+2)*N.y*N.z, devE );
	initMainArrays ( (N.x+2)*N.y*N.z, devH );
	//initMainArrays ( (N.x+1)*N.y*N.z, devCE );
	
	initPsiArrays ( Ntotpmlx, Db_pmlx, Dg_pmlx, psixE.y, psixE.z );
	initPsiArrays ( Ntotpmly, Db_pmly, Dg_pmly, psiyE.z, psiyE.x );
	initPsiArrays ( Ntotpmlz, Db_pmlz, Dg_pmlz, psizE.x, psizE.y );
	initPsiArrays ( Ntotpmlx, Db_pmlx, Dg_pmlx, psixH.y, psixH.z );
	initPsiArrays ( Ntotpmly, Db_pmly, Dg_pmly, psiyH.z, psiyH.x );
	initPsiArrays ( Ntotpmlz, Db_pmlz, Dg_pmlz, psizH.x, psizH.y );
	
	// --------------------------------------------------------------------------------
	// Copy arrays from host to device
	cudaMemcpy ( devCE.x, CE.x[0][0], size_devC, cudaMemcpyHostToDevice );
	cudaMemcpy ( devCE.y, CE.y[0][0], size_devC, cudaMemcpyHostToDevice );
	cudaMemcpy ( devCE.z, CE.z[0][0], size_devC, cudaMemcpyHostToDevice );

	free(CE.x);
	free(CE.y);
	free(CE.z);

	// --------------------------------------------------------------------------------
	// time loop
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++) {
		// E-fields main region update
		for ( i=0; i<NK; i++) updateE <<<Dg_main[i],Db_main,Ns_main>>> ( N, devE, devH, devCE, idx0[i] );
		
		// E-fields CPML region update
		updateCPMLxE <<<Dg_pmlx,Db_pmlx>>> ( N, devE, devH, devCE, psixE.y.f, psixE.z.f, 0);
		updateCPMLxE <<<Dg_pmlx,Db_pmlx>>> ( N, devE, devH, devCE, psixE.y.b, psixE.z.b, 1); 
		updateCPMLyE <<<Dg_pmly,Db_pmly>>> ( N, devE, devH, devCE, psiyE.z.f, psiyE.x.f, 0);
		updateCPMLyE <<<Dg_pmly,Db_pmly>>> ( N, devE, devH, devCE, psiyE.z.b, psiyE.x.b, 1);
		updateCPMLzE <<<Dg_pmlz,Db_pmlz,Ns_pmlz>>> ( N, devE, devH, devCE, psizE.x.f, psizE.y.f, 0);
		updateCPMLzE <<<Dg_pmlz,Db_pmlz,Ns_pmlz>>> ( N, devE, devH, devCE, psizE.x.b, psizE.y.b, 1);
			
		// Source update
		updateSrc <<<DGsrc,DBsrc>>> ( N, devE, tstep );

		// H-fields main region update
		for ( i=0; i<NK; i++) updateH <<<Dg_main[i],Db_main,Ns_main>>> ( N, devE, devH, idx0[i] );
		
		// H-fields CPML region update
		updateCPMLxH <<<Dg_pmlx,Db_pmlx>>> ( N, devE, devH, psixH.y.f, psixH.z.f, 0); 
		updateCPMLxH <<<Dg_pmlx,Db_pmlx>>> ( N, devE, devH, psixH.y.b, psixH.z.b, 1); 
		updateCPMLyH <<<Dg_pmlx,Db_pmlx>>> ( N, devE, devH, psiyH.z.f, psiyH.x.f, 0);
		updateCPMLyH <<<Dg_pmlx,Db_pmlx>>> ( N, devE, devH, psiyH.z.b, psiyH.x.b, 1);
		updateCPMLzH <<<Dg_pmlz,Db_pmlz,Ns_pmlz>>> ( N, devE, devH, psizH.x.f, psizH.y.f, 0);
		updateCPMLzH <<<Dg_pmlz,Db_pmlz,Ns_pmlz>>> ( N, devE, devH, psizH.x.b, psizH.y.b, 1);
			
		if ( tstep/50*50 == tstep ) {
			// Copy arrays from device to host
			cudaMemcpy( Ex[0][0], devE.x, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			//cudaMemcpy( Ez[0][0], devE.z, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			//cudaMemcpy( CEx[0][0], devCE.x, (N.x+1)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( Ey[0][0], devE.y, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( Ez[0][0], devE.z, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( Hx[0][0], devH.x, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( Hy[0][0], devH.y, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( Hz[0][0], devH.z, (N.x+2)*N.y*N.z*sizeof(float), cudaMemcpyDeviceToHost );

			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 1, N.x/2, N.y-1, N.z-2, Ex, "gpu_png/Ex-%05d.1.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.1.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 2, N.x/2, N.y-1, N.z-3, Ex, "gpu_png/Ex-%05d.2.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.2.h5", tstep);

			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ey, "gpu_png/Ey-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ey-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ez, "gpu_png/Ez-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ez-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Hx, "gpu_png/Hx-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Hx-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Hy, "gpu_png/Hy-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Hy-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Hz, "gpu_png/Hz-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Hz-%05d.h5", tstep);
			//dumpToH5(N.x+2, N.y, N.z, 0, 0, 0, N.x, N.y-1, 0, Ez, "gpu_png/Ez-%05d.h5", tstep);
			
			/*
			dumpToH5(N.x+2, N.y, N.z, 0, 0, N.z/2, N.x+1, N.y-1, N.z/2, Ex, "gpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, 0, 0, N.z/2, N.x+1, N.y-1, N.z/2, Ey, "gpu_png/Ey-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ey-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, 0, 0, N.z/2, N.x+1, N.y-1, N.z/2, Ez, "gpu_png/Ez-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ez-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, 0, 0, N.z/2, N.x+1, N.y-1, N.z/2, Hx, "gpu_png/Hx-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Hx-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, 0, 0, N.z/2, N.x+1, N.y-1, N.z/2, Hy, "gpu_png/Hy-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Hy-%05d.h5", tstep);
			dumpToH5(N.x+2, N.y, N.z, 0, 0, N.z/2, N.x+1, N.y-1, N.z/2, Hz, "gpu_png/Hz-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Hz-%05d.h5", tstep);
			*/
			
			//dumpToH5(N.x+1, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, CEx, "gpu_png/CEx-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/CEx-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
	
	free(Ex);
	free(Ey);
	free(Ez);
	free(Hx);
	free(Hy);
	free(Hz);
	freeMainArrays ( devE );
	freeMainArrays ( devH );
	freeMainArrays ( devCE );
	freePsiArrays ( psixE, psiyE, psizE );
	freePsiArrays ( psixH, psiyH, psizH );
}
