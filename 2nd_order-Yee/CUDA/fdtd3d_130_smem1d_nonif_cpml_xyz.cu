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
__constant__ float rcmbE[2*Npml];
__constant__ float rcmaE[2*Npml];
__constant__ float rcmbH[2*Npml];
__constant__ float rcmaH[2*Npml];


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


__host__ int selectTPB(int Nx, int Ny) {
	int Ntot = Nx*Ny;
	int TPB=1;

	if ( Ntot%32 == 0 ) TPB = 512;
	else if ( Ntot%16 == 0 ) TPB = 256;
	else if ( Ntot%8 == 0 ) TPB = 128;
	else printf("(%d,%d) mismatched TPB!\n", Nx, Ny);

	return TPB;
}


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


__global__ void initArray(int Ntot, float *a) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ( idx < Ntot ) a[idx] = 0;
}


__host__ void initMainArrays(N3 N, int Nzpit, P1F3 F) {
	int TPB=512;
	int Ntot = (N.x+1)*N.y*Nzpit;
	int BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
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


__global__ void updateE(int Nzpit, int Nyzpit, int TPB, P1F3 E, P1F3 H, P1F3 CE) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*TPB + tk;
	int eidx = idx + Nyzpit;

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[TPB+1];
	float* hz = (float*) &hy[TPB+1];

	hx[tk] = H.x[idx];
	hy[tk] = H.y[idx];
	hz[tk] = H.z[idx];
	if ( tk==TPB-1 ) {
		hx[tk+1] = H.x[idx+1];
		hy[tk+1] = H.y[idx+1];
	}
	__syncthreads();

	E.x[eidx] += CE.x[idx]*( H.z[idx+Nzpit] - hz[tk] - hy[tk+1] + hy[tk] );
	E.y[eidx] += CE.y[idx]*( hx[tk+1] - hx[tk] - H.z[idx+Nyzpit] + hz[tk] );
	E.z[eidx] += CE.z[idx]*( H.y[idx+Nyzpit] - hy[tk] - H.x[idx+Nzpit] + hx[tk] );
}


__global__ void updateH(int Nzpit, int Nyzpit, int TPB, P1F3 E, P1F3 H) {
	int tk = threadIdx.x;
	int idx = blockIdx.x*TPB + tk;
	int eidx = idx + Nyzpit;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[TPB+1];
	float* ez = (float*) &ey[TPB+1];

	ex[tk+1] = E.x[eidx];
	ey[tk+1] = E.y[eidx];
	ez[tk] = E.z[eidx];
	if ( tk==0 ) {
		ex[0] = E.x[eidx-1];
		ey[0] = E.y[eidx-1];
	}
	__syncthreads();

	H.x[idx] -= 0.5*( ez[tk] - E.z[eidx-Nzpit] - ey[tk+1] + ey[tk] );
	H.y[idx] -= 0.5*( ex[tk+1] - ex[tk] - ez[tk] + E.z[eidx-Nyzpit] );
	H.z[idx] -= 0.5*( ey[tk+1] - E.y[eidx-Nyzpit] - ex[tk+1] + E.x[eidx-Nzpit] );
}


__global__ void updateSrc(N3 N, int Nzpit, P1F3 E, int tstep) {
	int idx = threadIdx.x;
	int ijk = idx*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + (N.z/2);
	//int ijk = (N.x/2+1)*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + idx;

	E.x[ijk] += sin(0.1*tstep);
	//E.z[ijk] += sin(0.1*tstep);
}


__global__ void updateCPMLxE(
		int Nx, int Nzpit, int Nyzpit, int TPB, 
		P1F3 E, P1F3 H, P1F3 CE, 
		float *psi1, float *psi2,
		int backward) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int pi = pidx/Nyzpit + backward*Npml;

	int idx = pidx + backward*(Nx-Npml-1)*Nyzpit;
	int eidx = idx + Nyzpit;

	psi1[pidx] = rcmbE[pi]*psi1[pidx] + rcmaE[pi]*( H.z[idx+Nyzpit] - H.z[idx] );
	E.y[eidx] -= CE.y[idx]*psi1[pidx];

	psi2[pidx] = rcmbE[pi]*psi2[pidx] + rcmaE[pi]*( H.y[idx+Nyzpit] - H.y[idx] );
	E.z[eidx] += CE.z[idx]*psi2[pidx]
}


__global__ void updateCPMLxH(
		int Nx, int Nzpit, int Nyzpit, int TPB, 
		P1F3 E, P1F3 H,  
		float *psi1, float *psi2,
		int backward) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int pi = pidx/Nyzpit + backward*Npml;

	int idx = pidx + backward*(Nx-Npml)*Nyzpit;
	int eidx = idx + Nyzpit;

	psi1[pidx] = rcmbH[pi]*psi1[pidx] + rcmaH[pi]*( E.z[eidx] - E.z[eidx-Nyzpit] );
	H.y[idx] += 0.5*psi1[pidx];

	psi2[pidx] = rcmbH[pi]*psi2[pidx] + rcmaH[pi]*( E.y[eidx] - E.y[eidx-Nyzpit] );
	H.z[idx] -= 0.5*psi2[pidx];
}


__global__ void updateCPMLyE(
		int Ny, int Nzpit, int Npmlzpit, int TPB, 
		P1F3 E, P1F3 H, P1F3 CE, 
		float *psi1, float *psi2,
		int backward) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int i = pidx/Npmlzpit;
	int pj = ( pidx - i*Npmlzpit )/Nzpit + backward*Npml;

	int idx = pidx + (i+backward)*(Ny-Npml)*Nzpit - backward*Nzpit;
	int eidx = idx + Ny*Nzpit;

	psi1[pidx] = rcmbE[pj]*psi1[pidx] + rcmaE[pj]*( H.x[idx+Nzpit] - H.x[idx] );
	E.z[eidx] -= CE.z[idx]*psi1[pidx];

	psi2[pidx] = rcmbE[pj]*psi2[pidx] + rcmaE[pj]*( H.z[idx+Nzpit] - H.z[idx] );
	E.x[eidx] += CE.x[idx]*psi2[pidx];
}


__global__ void updateCPMLyH(
		int Ny, int Nzpit, int Npmlzpit, int TPB, 
		P1F3 E, P1F3 H, 
		float *psi1, float *psi2,
		int backward) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int i = pidx/Npmlzpit;
	int pj = ( pidx - i*Npmlzpit )/Nzpit + backward*Npml;

	int idx = pidx + (i+backward)*(Ny-Npml)*Nzpit;
	int eidx = idx + Ny*Nzpit;

	psi1[pidx] = rcmbH[pj]*psi1[pidx] + rcmaH[pj]*( E.x[eidx] - E.x[eidx-Nzpit] );
	H.z[idx] += 0.5*psi1[pidx];

	psi2[pidx] = rcmbH[pj]*psi2[pidx] + rcmaH[pj]*( E.z[eidx] - E.z[eidx-Nzpit] );
	H.x[idx] -= 0.5*psi2[pidx];
}


__global__ void updateCPMLzE(
		N3 N, int Nzpit, int TPB,
		P1F3 E, P1F3 H, P1F3 CE,
		float *psi1, float *psi2,
		int backward) {
	int tk = threadIdx.x;
	int pidx = blockIdx.x*TPB + tk;
	int Npmlp = Npml+1;
	int i = pidx/(N.y*Npmlp);
	int j = ( pidx - i*N.y*Npmlp )/Npmlp;
	int pk = pidx - i*N.y*Npmlp - j*Npmlp;// + backward*Npml;

	int idx = pidx + (j+i*N.y)*(Nzpit-Npmlp);// + backward*(N.z-Npml-1);
	int eidx = idx + N.y*Nzpit;

	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[TPB+1];

	hx[tk] = H.x[idx];
	hy[tk] = H.y[idx];
	if ( tk==TPB-1 ) {
		hx[tk+1] = H.x[idx+1];
		hy[tk+1] = H.y[idx+1];
	}
	__syncthreads();

	if ( i<N.x-1 && pk<Npml ) {
		psi1[pidx] = rcmbE[pk]*psi1[pidx] + rcmaE[pk]*( hy[tk+1] - hy[tk] );
		E.x[eidx] -= CE.x[idx]*psi1[pidx];

		psi2[pidx] = rcmbE[pk]*psi2[pidx] + rcmaE[pk]*( hx[tk+1] - hx[tk] );
		E.y[eidx] += CE.y[idx]*psi2[pidx];
	}
}


__global__ void updateCPMLzH(
		N3 N, int Nzpit, int TPB,
		P1F3 E, P1F3 H,
		float *psi1, float *psi2,
		int backward) {
	int tk = threadIdx.x;
	int pidx = blockIdx.x*TPB + tk;
	int Npmlp = Npml+1;
	int i = pidx/(N.y*Npmlp);
	int j = ( pidx - i*N.y*Npmlp )/Npmlp;
	int pk = pidx - i*N.y*Npmlp - j*Npmlp;// + backward*Npml;

	int idx = pidx + (j+i*N.y)*(Nzpit-Npmlp);// + backward*(N.z-Npml-1);
	int eidx = idx + N.y*Nzpit;

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[TPB+1];

	//printf("[%.3d,%.3d,%.3d]\t\t%d\t\t%d\t\t%d\n",i,j,pk,pidx,idx,eidx);
	if ( i<N.x ) {
		ex[tk+1] = E.x[eidx];
		ey[tk+1] = E.y[eidx];
		if ( tk==0 ) {
			ex[0] = E.x[eidx-1];
			ey[0] = E.y[eidx-1];
		}
	}
	__syncthreads();

	if ( i>0 && i<N.x && pk<Npml ) {
		psi1[pidx] = rcmbH[pk]*psi1[pidx] + rcmaH[pk]*( ey[tk+1] - ey[tk] );
		H.x[idx] += 0.5*psi1[pidx];

		psi2[pidx] = rcmbH[pk]*psi2[pidx] + rcmaH[pk]*( ex[tk+1] - ex[tk] );
		H.y[idx] -= 0.5*psi2[pidx];
	}
}


__global__ void init_boundary_xE(N3 N, int Nyzpit, P1F3 E) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int eidx = idx + N.x*Nyzpit;

	if ( idx/Nyzpit == 0 ) {
		E.y[eidx] = 0;
		E.z[eidx] = 0;
	}
}


__global__ void init_boundary_yE(N3 N, int Nzpit, int Nyzpit, P1F3 E) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx/Nzpit;
	int k = idx - i*Nzpit;
	int eidx = (i+1)*Nyzpit + (N.y-1)*Nzpit + k;

	if ( i<N.x ) {
		E.z[eidx] = 0;
		E.x[eidx] = 0;
	}
}


__global__ void init_boundary_zE(N3 N, int Nzpit, int Nyzpit, P1F3 E) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx/N.y;
	int j = idx - i*N.y;
	int eidx = (i+1)*Nyzpit + j*Nzpit + (N.z-1);

	if ( i<N.x ) {
		E.x[eidx] = 0;
		E.y[eidx] = 0;
	}
}


__global__ void init_boundary_xH(N3 N, int Nyzpit, P1F3 H) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ( idx/Nyzpit == 0 ) {
		H.x[idx] = 0;
		H.y[idx] = 0;
		H.z[idx] = 0;
	}
}


__global__ void init_boundary_yH(N3 N, int Nzpit, int Nyzpit, P1F3 H) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx/Nzpit;
	int k = idx - i*Nzpit;
	int hidx = i*Nyzpit + k;

	if ( i<N.x ) {
		H.x[hidx] = 0;
		H.y[hidx] = 0;
		H.z[hidx] = 0;
	}
}


__global__ void init_boundary_zH(N3 N, int Nzpit, int Nyzpit, P1F3 H) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx/N.y;
	int j = idx - i*N.y;
	int hidx = i*Nyzpit + j*Nzpit;

	//int i0 = idx/Nyzpit;
	//int j0 = (idx - i0*Nyzpit)/Nzpit;
	//int k0 = idx - i0*Nyzpit - j0*Nzpit;
	//printf("[%d,%d,%d] %d\n", i0,j0,k0,idx);

	if ( i<N.x ) {
		//printf("\t\t\tIn: [%d,%d,%d] %d\n", i0,j0,k0,idx);
		H.x[hidx] = 0;
		H.y[hidx] = 0;
		H.z[hidx] = 0;
	}
}


__host__ void update_boundary_E(N3 N, int Nzpit, int Nyzpit, P1F3 devE) {
	int Ntot, BPG;
	int TPB = 128;	 

	/*
	Ntot = Nyzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	init_boundary_xE <<<dim3(BPG),dim3(TPB)>>> ( N, Nyzpit, devE );
	
	Ntot = N.x*Nzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	init_boundary_yE <<<dim3(BPG),dim3(TPB)>>> ( N, Nzpit, Nyzpit, devE );
	*/

	Ntot = N.x*N.y;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	init_boundary_zE <<<dim3(BPG),dim3(TPB)>>> ( N, Nzpit, Nyzpit, devE );
}


__host__ void update_boundary_H(N3 N, int Nzpit, int Nyzpit, P1F3 devH) {
	int Ntot, BPG;
	int TPB = 128;	 

	/*
	Ntot = Nyzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	init_boundary_xH <<<dim3(BPG),dim3(TPB)>>> ( N, Nyzpit, devH );
	
	Ntot = N.x*Nzpit;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	init_boundary_yH <<<dim3(BPG),dim3(TPB)>>> ( N, Nzpit, Nyzpit, devH );
	*/
	
	Ntot = N.x*N.y;
	BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1;
	init_boundary_zH <<<dim3(BPG),dim3(TPB)>>> ( N, Nzpit, Nyzpit, devH );
}


int main() {
	int tstep;
	char time_str[32];
	time_t t0;
	int i;

	// Set the parameters
	N3 N;
	N.x = 200;
	N.y = 200;
	N.z = 200;
	//N.y = 16;
	//N.z = 20;
	int TMAX = 10000;

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
	N3 Nxp;
	Nxp.x = N.x+1;
	Nxp.y = N.y;
	Nxp.z = N.z;
	Ex = makeArray(Nxp);
	Ez = makeArray(Nxp);

	// Geometry
	set_geometry(N, CE);

	// CPML
	int m = 4;	// grade_order
	float sigma_max = (m+1.)/(15*pi*Npml*dx);
	float alpha = 0.05;
	float *sigmaE, *bE, *aE;
	float *sigmaH, *bH, *aH;

	sigmaE = (float *) calloc (2*Npml, sizeof(float));
	sigmaH = (float *) calloc (2*Npml, sizeof(float));
	bE = (float *) calloc (2*Npml, sizeof(float));
	bH = (float *) calloc (2*Npml, sizeof(float));
	aE = (float *) calloc (2*Npml, sizeof(float));
	aH = (float *) calloc (2*Npml, sizeof(float));
	for (i=0; i<Npml; i++) {
		sigmaE[i] = pow( (Npml-0.5-i)/Npml, m )*sigma_max;
		sigmaE[i+Npml] = pow( (0.5+i)/Npml, m )*sigma_max;
		sigmaH[i] = pow( (float)(Npml-i)/Npml, m )*sigma_max;
		sigmaH[i+Npml] = pow( (1.+i)/Npml, m )*sigma_max;
	}

	for (i=0; i<2*Npml; i++) {
		bE[i] = exp( -(sigmaE[i] + alpha)*dt/ep0 );
		bH[i] = exp( -(sigmaH[i] + alpha)*dt/ep0 );
		aE[i] = sigmaE[i]/(sigmaE[i]+alpha)*(bE[i]-1);
		aH[i] = sigmaH[i]/(sigmaH[i]+alpha)*(bH[i]-1);
		//printf("[%d]\tsigmaE=%g,\tbE=%g,aE=%g\n", i, sigmaE[i], bE[i], aE[i]);
		//printf("[%d]\tsigmaH=%g,\tbH=%g,aH=%g\n", i, sigmaH[i], bH[i], aH[i]);
	}
	free(sigmaE);
	free(sigmaH);

	// Copy arrays from host to constant memory
	cudaMemcpyToSymbol(rcmbE, bE, 2*Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmaE, aE, 2*Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmbH, bH, 2*Npml*sizeof(float));
	cudaMemcpyToSymbol(rcmaH, aH, 2*Npml*sizeof(float));

	// Allocate device memory
	P1F3 devE, devH;
	P1F3 devCE;
	int z_size = N.z*sizeof(float);
	size_t pitch;
	cudaMallocPitch ( (void**) &devE.x, &pitch, z_size, (N.x+1)*N.y );
	cudaMallocPitch ( (void**) &devE.y, &pitch, z_size, (N.x+1)*N.y );
	cudaMallocPitch ( (void**) &devE.z, &pitch, z_size, (N.x+1)*N.y );
	cudaMallocPitch ( (void**) &devH.x, &pitch, z_size, (N.x+1)*N.y );
	cudaMallocPitch ( (void**) &devH.y, &pitch, z_size, (N.x+1)*N.y );
	cudaMallocPitch ( (void**) &devH.z, &pitch, z_size, (N.x+1)*N.y );
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
	cudaMallocPitch ( (void**) &psixH.y.f, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixH.y.b, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixH.z.f, &pitch, z_size, Npml*N.y );
	cudaMallocPitch ( (void**) &psixH.z.b, &pitch, z_size, Npml*N.y );

	cudaMallocPitch ( (void**) &psiyE.z.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyE.z.b, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyE.x.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyE.x.b, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.z.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.z.b, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.x.f, &pitch, z_size, N.x*Npml );
	cudaMallocPitch ( (void**) &psiyH.x.b, &pitch, z_size, N.x*Npml );

	cudaMalloc ( (void**) &psizE.x.f, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizE.x.b, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizE.y.f, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizE.y.b, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizH.x.f, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizH.x.b, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizH.y.f, N.x*N.y*(Npml+1)*sizeof(float) );
	cudaMalloc ( (void**) &psizH.y.b, N.x*N.y*(Npml+1)*sizeof(float) );

	// Copy arrays from host to device
	cudaMemcpy2D ( devCE.x, pitch, CE.x[0][0], z_size, z_size, N.x*N.y, cudaMemcpyHostToDevice );
	cudaMemcpy2D ( devCE.y, pitch, CE.y[0][0], z_size, z_size, N.x*N.y, cudaMemcpyHostToDevice );
	cudaMemcpy2D ( devCE.z, pitch, CE.z[0][0], z_size, z_size, N.x*N.y, cudaMemcpyHostToDevice );

	free(CE.x);
	free(CE.y);
	free(CE.z);

	int Nz_pitch = pitch/4;
	printf("pitch= %u, Nz_pitch= %d\n", pitch, Nz_pitch);

	// Set the GPU parameters
	// TPB: Number of threads per block
	// BPG: Number of thread blocks per grid
	int Ntot = N.x*N.y*Nz_pitch;
	int TPBmain = selectTPB(N.x, N.y);	 
	int BPGmain = Ntot%TPBmain == 0 ? Ntot/TPBmain : Ntot/TPBmain + 1;
	dim3 Dg = dim3(BPGmain);
	dim3 Db = dim3(TPBmain);
	size_t Ns = sizeof(float)*( 2*(TPBmain+1)+(TPBmain) );
	printf("Threads per block: %d\n", TPBmain);
	printf("Blocks per grid: %d\n", BPGmain);
	verify_over_TPB( TPBmain );
	verify_over_BPG( BPGmain );
	printf("Number of bytes in shared memory: %d\n", Ns);

	int TPBsrc = N.x;
	//int TPBsrc = N.z;
	int BPGsrc = 1;
	dim3 Dgsrc(BPGsrc);
	dim3 Dbsrc(TPBsrc);

	int TPBpmlx = selectTPB(Npml, N.y);	 
	int Ntotpmlx = Npml*N.y*Nz_pitch;
	int BPGpmlx = Ntotpmlx%TPBpmlx == 0 ? Ntotpmlx/TPBpmlx : Ntotpmlx/TPBpmlx + 1;
	dim3 Dgpmlx(BPGpmlx);
	dim3 Dbpmlx(TPBpmlx);
	printf("CPMLx: Threads per block: %d\n", TPBpmlx);
	printf("CPMLx: Blocks per grid: %d\n", BPGpmlx);
	verify_over_BPG( BPGpmlx );

	int TPBpmly = selectTPB(N.x, Npml);
	int Ntotpmly = N.x*Npml*Nz_pitch;
	int BPGpmly = Ntotpmly%TPBpmly == 0 ? Ntotpmly/TPBpmly : Ntotpmly/TPBpmly + 1;
	dim3 Dgpmly(BPGpmly);
	dim3 Dbpmly(TPBpmly);
	printf("CPMLy: Threads per block: %d\n", TPBpmly);
	printf("CPMLy: Blocks per grid: %d\n", BPGpmly);
	verify_over_BPG( BPGpmly );

	int TPBpmlz = selectTPB(N.x, N.y);
	int Ntotpmlz = N.x*N.y*(Npml+1);
	int BPGpmlz = Ntotpmlz%TPBpmlz == 0 ? Ntotpmlz/TPBpmlz : Ntotpmlz/TPBpmlz + 1;
	dim3 Dgpmlz(BPGpmlz);
	dim3 Dbpmlz(TPBpmlz);
	printf("CPMLz: Threads per block: %d\n", TPBpmlz);
	printf("CPMLz: Blocks per grid: %d\n", BPGpmlz);
	verify_over_BPG( BPGpmlz );
	size_t Nspmlz = sizeof(float)*( 2*(TPBpmlz+1) );

	// Initialize the device arrays
	initMainArrays ( N, Nz_pitch, devE );
	initMainArrays ( N, Nz_pitch, devH );
	initPsiArrays ( N, Nz_pitch, psixE, psiyE, psizE );
	initPsiArrays ( N, Nz_pitch, psixH, psiyH, psizH );

	// Main time loop
	t0 = time(0);
	//for ( tstep=1; tstep<=TMAX; tstep++) {
	for ( tstep=1; tstep<=100; tstep++) {
		// Update on the GPU
		updateE <<<Dg,Db,Ns>>> ( Nz_pitch, N.y*Nz_pitch, TPBmain, devE, devH, devCE );
		update_boundary_E(N, Nz_pitch, N.y*Nz_pitch, devE); 
		updateCPMLxE <<<Dgpmlx,Dbpmlx>>> ( N.x, Nz_pitch, N.y*Nz_pitch, TPBpmlx, devE, devH, devCE, psixE.y.f, psixE.z.f, 0); 
		updateCPMLxE <<<Dgpmlx,Dbpmlx>>> ( N.x, Nz_pitch, N.y*Nz_pitch, TPBpmlx, devE, devH, devCE, psixE.y.b, psixE.z.b, 1); 
		updateCPMLyE <<<Dgpmly,Dbpmly>>> ( N.y, Nz_pitch, Npml*Nz_pitch, TPBpmly, devE, devH, devCE, psiyE.z.f, psiyE.x.f, 0); 
		updateCPMLyE <<<Dgpmly,Dbpmly>>> ( N.y, Nz_pitch, Npml*Nz_pitch, TPBpmly, devE, devH, devCE, psiyE.z.b, psiyE.x.b, 1); 
		updateCPMLzE <<<Dgpmlz,Dbpmlz,Nspmlz>>> ( N, Nz_pitch, TPBpmlz, devE, devH, devCE, psizE.x.f, psizE.y.f, 0); 


		updateSrc <<<Dgsrc,Dbsrc>>> ( N, Nz_pitch, devE, tstep );

		updateH <<<Dg,Db,Ns>>> ( Nz_pitch, N.y*Nz_pitch, TPBmain, devE, devH );
		update_boundary_H(N, Nz_pitch, N.y*Nz_pitch, devH); 
		updateCPMLxH <<<Dgpmlx,Dbpmlx>>> ( N.x, Nz_pitch, N.y*Nz_pitch, TPBpmlx, devE, devH, psixH.y.f, psixH.z.f, 0); 
		updateCPMLxH <<<Dgpmlx,Dbpmlx>>> ( N.x, Nz_pitch, N.y*Nz_pitch, TPBpmlx, devE, devH, psixH.y.b, psixH.z.b, 1); 
		updateCPMLyH <<<Dgpmly,Dbpmly>>> ( N.y, Nz_pitch, Npml*Nz_pitch, TPBpmly, devE, devH, psiyH.z.f, psiyH.x.f, 0); 
		updateCPMLyH <<<Dgpmly,Dbpmly>>> ( N.y, Nz_pitch, Npml*Nz_pitch, TPBpmly, devE, devH, psiyH.z.b, psiyH.x.b, 1); 
		//updateCPMLzH <<<Dgpmlz,Dbpmlz,Nspmlz>>> ( N, Nz_pitch, TPBpmlz, devE, devH, psizH.x.f, psizH.y.f, 0); 


		if ( tstep/10*10 == tstep ) {
			// Copy arrays from device to host
			cudaMemcpy2D( Ex[0][0], z_size, devE.x, pitch, z_size, (N.x+1)*N.y, cudaMemcpyDeviceToHost );
			//cudaMemcpy2D( Ez[0][0], z_size, devE.z, pitch, z_size, (N.x+1)*N.y, cudaMemcpyDeviceToHost );

			//print_array(N, Ex);
			dumpToH5(N.x+1, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);
			//dumpToH5(N.x+1, N.y, N.z, 0, 0, N.z/2, N.x, N.y-1, N.z/2, Ez, "gpu_png/Ez-%05d.h5", tstep);
			//dumpToH5(N.x+1, N.y, N.z, 0, 0, 0, N.x, N.y-1, 0, Ez, "gpu_png/Ez-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ez-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
	//update_boundary_E(N, Nz_pitch, N.y*Nz_pitch, devE); 
	//update_boundary_H(N, Nz_pitch, N.y*Nz_pitch, devH); 
	//for ( tstep=1; tstep<=10; tstep++ ) updateE <<<Dg,Db,Ns>>> ( Nz_pitch, N.y*Nz_pitch, TPBmain, devE, devH, devCE );
	//for ( tstep=1; tstep<=10; tstep++ ) updateH <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPBmain, devE, devH );
	
	//for ( tstep=1; tstep<=10; tstep++ )	updateCPMLxE <<<Dgpmlx,Dbpmlx>>> ( N.x, Nz_pitch, N.y*Nz_pitch, TPBpmlx, devE, devH, devCE, psixE.y.f, psixE.z.f, 0); 
	//for ( tstep=1; tstep<=10; tstep++ )	updateCPMLxE <<<Dgpmlx,Dbpmlx>>> ( N.x, Nz_pitch, N.y*Nz_pitch, TPBpmlx, devE, devH, devCE, psixE.y.b, psixE.z.b, 1); 
	
	free(Ex);
	free(Ez);
	freeMainArrays ( devE );
	freeMainArrays ( devH );
	freeMainArrays ( devCE );
	freePsiArrays ( psixE, psiyE, psizE );
	freePsiArrays ( psixH, psiyH, psizH );
}
