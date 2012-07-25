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


__host__ void initPsiArrays(N3 N, int Nzpit, int Npmlpit, P1F6 psix, P1F6 psiy, P1F6 psiz) {
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

	Ntot = N.x*N.y*Npmlpit;
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


__global__ void updateSrc(N3 N, int Nzpit, P1F3 E, int tstep) {
	int idx, ijk;

	idx = blockIdx.x*blockDim.x + threadIdx.x;
	ijk = idx*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + (N.z/2);
	//ijk = (N.x/2)*(N.y)*(Nzpit) + (N.y/2)*(Nzpit) + idx;

	if ( idx < N.x ) E.x[ijk] += sin(0.1*tstep);
	//if ( idx < N.z ) E.z[ijk] += sin(0.1*tstep);
}


__global__ void updateCPMLx(
		N3 N, int Nzpit, int TPB, 
		P1F3 E, P1F3 H, P1F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float *psi1, float *psi2,
		int EorH, int backward) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int area = N.y*Nzpit;
	int pi = pidx/area;
	int j = ( pidx - pi*area )/Nzpit;
	int k = pidx - pi*area - j*Nzpit;

	int idx = pidx + backward*(N.x-Npml-EorH)*area;

	if ( pi>is && pi<ie && j>js && j<je && k>ks && k<ke ) {
		int i = pi + backward*Npml;
		if ( EorH ) {
			psi1[pidx] = rcmbE[i]*psi1[pidx] + rcmaE[i]*( H.z[idx+area] - H.z[idx] );
			E.y[idx] -= CE.y[idx]*psi1[pidx];

			psi2[pidx] = rcmbE[i]*psi2[pidx] + rcmaE[i]*( H.y[idx+area] - H.y[idx] );
			E.z[idx] += CE.z[idx]*psi2[pidx];
		}
		else {
			psi1[pidx] = rcmbH[i]*psi1[pidx] + rcmaH[i]*( E.z[idx] - E.z[idx-area] );
			H.y[idx] += 0.5*psi1[pidx];

			psi2[pidx] = rcmbH[i]*psi2[pidx] + rcmaH[i]*( E.y[idx] - E.y[idx-area] );
			H.z[idx] -= 0.5*psi2[pidx];
		}
	}
}


__global__ void updateCPMLy(
		N3 N, int Nzpit, int TPB, 
		P1F3 E, P1F3 H, P1F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float *psi1, float *psi2,
		int EorH, int backward) {
	int pidx = blockIdx.x*TPB + threadIdx.x;
	int area = Npml*Nzpit;
	int i = pidx/area;
	int pj = ( pidx - i*area )/Nzpit;
	int k = pidx - i*area - pj*Nzpit;

	int idx = pidx + (i+backward)*(N.y-Npml)*Nzpit - backward*EorH*Nzpit;

	if ( i>is && i<ie && pj>js && pj<je && k>ks && k<ke ) {
		int i = pj + backward*Npml;
		if ( EorH ) {
			psi1[pidx] = rcmbE[i]*psi1[pidx] + rcmaE[i]*( H.x[idx+Nzpit] - H.x[idx] );
			E.z[idx] -= CE.z[idx]*psi1[pidx];

			psi2[pidx] = rcmbE[i]*psi2[pidx] + rcmaE[i]*( H.z[idx+Nzpit] - H.z[idx] );
			E.x[idx] += CE.x[idx]*psi2[pidx];
		}
		else {
			psi1[pidx] = rcmbH[i]*psi1[pidx] + rcmaH[i]*( E.x[idx] - E.x[idx-Nzpit] );
			H.z[idx] += 0.5*psi1[pidx];

			psi2[pidx] = rcmbH[i]*psi2[pidx] + rcmaH[i]*( E.z[idx] - E.z[idx-Nzpit] );
			H.x[idx] -= 0.5*psi2[pidx];
		}
	}
}


__global__ void updateCPMLzE(
		N3 N, int Nzpit, int Npmlpit, 
		P1F3 E, P1F3 H, P1F3 CE,
		float *psi1, float *psi2,
		int backward) {
	int pidx = blockIdx.x*Npmlpit + threadIdx.x;
	int area = N.y*Npmlpit;
	int i = pidx/area;
	int j = ( pidx - i*area )/Npmlpit;
	int pk = pidx - i*area - j*Npmlpit;

	int idx = pidx + (j+i*N.y)*(Nzpit-Npmlpit) + backward*(N.z-Npml-1);

	//printf("[%.5d]\t[%.5d]\t[%.3d, %.3d, %.3d]\n", pidx, idx, i, j, pk);
	extern __shared__ float hs[];
	float* hx = (float*) hs;
	float* hy = (float*) &hx[Npml+1];

	if ( i<N.x-1 && j<N.y-1 && pk<Npml ) {
		hx[pk] = H.x[idx];
		hy[pk] = H.y[idx];
		if ( pk==Npml-1 ) {
			hx[pk+1] = H.x[idx+1];
			hy[pk+1] = H.y[idx+1];
		}
	}
	__syncthreads();

	if ( i<N.x-1 && j<N.y-1 && pk<Npml ) {
		int i = pk + backward*Npml;
		psi1[pidx] = rcmbE[i]*psi1[pidx] + rcmaE[i]*( hy[pk+1] - hy[pk] );
		E.x[idx] -= CE.x[idx]*psi1[pidx];

		psi2[pidx] = rcmbE[i]*psi2[pidx] + rcmaE[i]*( hx[pk+1] - hx[pk] );
		E.y[idx] += CE.y[idx]*psi2[pidx];
	}
}


__global__ void updateCPMLzH(
		N3 N, int Nzpit, int Npmlpit, 
		P1F3 E, P1F3 H, 
		float *psi1, float *psi2,
		int backward) {
	int pidx = blockIdx.x*Npmlpit + threadIdx.x;
	int area = N.y*Npmlpit;
	int i = pidx/area;
	int j = ( pidx - i*area )/Npmlpit;
	int pk = pidx - i*area - j*Npmlpit;

	int idx = pidx + (j+i*N.y)*(Nzpit-Npmlpit) + backward*(N.z-Npml);

	extern __shared__ float es[];
	float* ex = (float*) es;
	float* ey = (float*) &ex[Npml+1];

	if ( i>0 && i<N.x && j>0 && pk>-backward && pk<Npml ) {
		ex[pk+1] = E.x[idx];
		ey[pk+1] = E.y[idx];
		if ( pk==0 ) {
			ex[0] = E.x[idx-1];
			ey[0] = E.y[idx-1];
		}
	}
	__syncthreads();

	if ( i>0 && i<N.x && j>0 && pk>-backward && pk<Npml ) {
		int i = pk + backward*Npml;
		psi1[pidx] = rcmbH[i]*psi1[pidx] + rcmaH[i]*( ey[pk+1] - ey[pk] );
		H.x[idx] += 0.5*psi1[pidx];

		psi2[pidx] = rcmbH[i]*psi2[pidx] + rcmaH[i]*( ex[pk+1] - ex[pk] );
		H.y[idx] -= 0.5*psi2[pidx];
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
	N.y = 200;
	N.z = 500;
	//N.x = 30;
	//N.y = 30;
	//N.z = 40;
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
	Ex = makeArray(N);
	Ez = makeArray(N);

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
	int TPB = 512;	// Number of threads per block
	int BPG = Ntot%TPB == 0 ? Ntot/TPB : Ntot/TPB + 1; // Number of thread blocks per grid
	dim3 Dg = dim3(BPG);
	dim3 Db = dim3(TPB);
	size_t Ns = sizeof(float)*( (TPB+1)+(TPB+1)+(TPB) );
	printf("Threads per block: %d\n", TPB);
	printf("Blocks per grid: %d\n", BPG);
	verify_over_TPB( TPB );
	verify_over_BPG( BPG );
	printf("Number of bytes in shared memory: %d\n", Ns);

	int TPBsrc = N.x;
	//int TPBsrc = N.z;
	int BPGsrc = 1;
	dim3 Dgsrc(BPGsrc);
	dim3 Dbsrc(TPBsrc);
	verify_over_TPB( TPBsrc );
	verify_over_BPG( BPGsrc );

	int TPBpmlx = 512;
	int Ntotpmlx = Npml*N.y*Nz_pitch;
	int BPGpmlx = Ntotpmlx%TPBpmlx == 0 ? Ntotpmlx/TPBpmlx : Ntotpmlx/TPBpmlx + 1;
	dim3 Dgpmlx(BPGpmlx);
	dim3 Dbpmlx(TPBpmlx);
	verify_over_BPG( BPGpmlx );

	int TPBpmly = 512;
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
	initPsiArrays ( N, Nz_pitch, Nzpml_pitch, psixE, psiyE, psizE );
	initPsiArrays ( N, Nz_pitch, Nzpml_pitch, psixH, psiyH, psizH );

	// Main time loop
	t0 = time(0);
	int elapsedTime;
	//for ( tstep=1; tstep<=TMAX; tstep++) {
	for ( tstep=1; tstep<=1; tstep++) {
		// Update on the GPU
		updateE <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH, devCE );
		updateCPMLx <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, -1, Npml, -1, N.y-1, -1, N.z-1, psixE.y.f, psixE.z.f, 1, 0);
		updateCPMLx <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, -1, Npml, -1, N.y-1, -1, N.z-1, psixE.y.b, psixE.z.b, 1, 1);
		updateCPMLy <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, -1, N.x-1, -1, Npml, -1, N.z-1, psiyE.z.f, psiyE.x.f, 1, 0);
		updateCPMLy <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, -1, N.x-1, -1, Npml, -1, N.z-1, psiyE.z.b, psiyE.x.b, 1, 1);
		updateCPMLzE <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, devCE, psizE.x.f, psizE.y.f, 0);
		updateCPMLzE <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, devCE, psizE.x.b, psizE.y.b, 1);

		updateSrc <<<Dgsrc,Dbsrc>>> ( N, Nz_pitch, devE, tstep );

		updateH <<<Dg,Db,Ns>>> ( N, Nz_pitch, TPB, devE, devH );
		updateCPMLx <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE,  0, Npml, 0, N.y, 0, N.z, psixH.y.f, psixH.z.f, 0, 0);
		updateCPMLx <<<Dgpmlx,Dbpmlx>>> (N, Nz_pitch, TPBpmlx, devE, devH, devCE, -1, Npml, 0, N.y, 0, N.z, psixH.y.b, psixH.z.b, 0, 1);
		updateCPMLy <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, 0, N.x,  0, Npml, 0, N.z, psiyH.z.f, psiyH.x.f, 0, 0);
		updateCPMLy <<<Dgpmly,Dbpmly>>> (N, Nz_pitch, TPBpmly, devE, devH, devCE, 0, N.x, -1, Npml, 0, N.z, psiyH.z.b, psiyH.x.b, 0, 1);
		updateCPMLzH <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, psizH.x.f, psizH.y.f, 0);
		updateCPMLzH <<<Dgpmlz,Dbpmlz,Nspmlz>>> (N, Nz_pitch, Nzpml_pitch, devE, devH, psizH.x.b, psizH.y.b, 1);
		
		if ( tstep/1000*1000 == tstep ) {
			// Copy arrays from device to host
			//cudaMemcpy2D( Ex[0][0], z_size, devE.x, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );
			//cudaMemcpy2D( Ez[0][0], z_size, devE.z, pitch, z_size, N.x*N.y, cudaMemcpyDeviceToHost );

			//print_array(N, Ex);
			//dumpToH5(N.x, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, Ex, "gpu_png/Ex-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ex-%05d.h5", tstep);
			//dumpToH5(N.x, N.y, N.z, 0, 0, N.z/2, N.x-1, N.y-1, N.z/2+1, Ez, "gpu_png/Ez-%05d.h5", tstep);
			//dumpToH5(N.x, N.y, N.z, 0, 0, 0, N.x-1, N.y-1, 1, Ez, "gpu_png/Ez-%05d.h5", tstep);
			//dumpToH5(N.x, N.y, N.z, 0, 0, N.z-2, N.x-1, N.y-1, N.z-1, Ez, "gpu_png/Ez-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered gpu_png/Ez-%05d.h5", tstep);

			//updateTimer(t0, tstep, time_str);
			//printf("tstep=%d\t%s\n", tstep, time_str);
			elapsedTime=(int)(time(0)-t0);
			printf("tstep=%d\t%d\n", tstep, elapsedTime);
		}
	}
	//updateTimer(t0, tstep, time_str);
	//printf("tstep=%d\t%s\n", tstep, time_str);

	freeMainArrays ( devE );
	freeMainArrays ( devH );
	freePsiArrays ( psixE, psiyE, psizE );
	freePsiArrays ( psixH, psiyH, psizH );
}
