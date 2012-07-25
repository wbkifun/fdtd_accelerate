#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>
#include <omp.h>

const float light_velocity = 2.99792458e8;	// m s- 

#define LOAD __builtin_ia32_loadups
#define STORE __builtin_ia32_storeups
#define ADD __builtin_ia32_addps
#define SUB __builtin_ia32_subps
#define MUL __builtin_ia32_mulps
typedef float v4sf __attribute__ ((vector_size(16)));


typedef struct N3 {
	int x, y, z;
} N3;


typedef struct P3F3 {
	float ***x, ***y, ***z;
} P3F3;



void updateTimer(time_t t0, int tstep, char str[]) {
	int elapsedTime=(int)(time(0)-t0);
	sprintf(str, "%02d:%02d:%02d (%d)", elapsedTime/3600, elapsedTime%3600/60, elapsedTime%60, elapsedTime);
}


void exec(char *format, ...) {
	char str[1024];
	va_list ap;
	va_start(ap, format);
	vsprintf(str, format, ap);
	system(str);
}


void dumpToH5(int Ni, int Nj, int Nk, int is, int js, int ks, int ie, int je, int ke, float ***f, char *format, ...) {
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


float ***makeArray3D(int Nx, int Ny, int Nz, int add) {
	float ***f;
	int i;

	f = (float ***) calloc (Nx, sizeof(float **));
	f[0] = (float **) calloc (Ny*Nx, sizeof(float *));
	f[0][0] = (float *) calloc (Nz*Ny*Nx + add, sizeof(float));

	for (i=0; i<Nx; i++) f[i] = f[0] + i*Ny;
	for (i=0; i<Ny*Nx; i++) f[0][i] = f[0][0] + i*Nz;

	return f;
}


void set_geometry( N3 N, P3F3 CE ) {
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
}


void updateE( int Ntot, int Ny, int Nz, 
		float *Ex, float *Ey, float *Ez,
		float *Hx, float *Hy, float *Hz,
		float *CEx, float *CEy, float *CEz ) {
	int idx, i;
	int Nyz = Ny*Nz;
	int c1 = (Ny-1)*Nz, c2 = Nyz + Nz;
	v4sf e, ce, h1, h2, h3, h4; 

	omp_set_num_threads(8);
	#pragma omp parallel for \
	shared( Ntot, Nz, Nyz, c1, c2, Ex, Ey, Ez, Hx, Hy, Hz, CEx, CEy, CEz ) \
   	private( e, ce, h1, h2, h3, h4, idx, i ) \
	schedule( static )
	for ( idx=0; idx<Ntot; idx+=4 ) {
		i = idx + idx/c1*Nz + c2;

		e  = LOAD( &Ex[i] );
		ce = LOAD( &CEx[i] );
		h1 = LOAD( &Hz[i+Nz] );
		h2 = LOAD( &Hz[i] );
		h3 = LOAD( &Hy[i+1] );
		h4 = LOAD( &Hy[i] );
		STORE( &Ex[i], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4)))) ); 

		e  = LOAD( &Ey[i] );
		ce = LOAD( &CEy[i] );
		h1 = LOAD( &Hx[i+1] );
		h2 = LOAD( &Hx[i] );
		h3 = LOAD( &Hz[i+Nyz] );
		h4 = LOAD( &Hz[i] );
		STORE( &Ey[i], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4)))) ); 

		e  = LOAD( &Ez[i] );
		ce = LOAD( &CEz[i] );
		h1 = LOAD( &Hy[i+Nyz] );
		h2 = LOAD( &Hy[i] );
		h3 = LOAD( &Hx[i+Nz] );
		h4 = LOAD( &Hx[i] );
		STORE( &Ez[i], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4)))) ); 
	}
}


void updateH( int Ntot, int Ny, int Nz, 
		float *Ex, float *Ey, float *Ez,
		float *Hx, float *Hy, float *Hz ) {
	int idx, i;
	int Nyz = Ny*Nz;
	int c1 = (Ny-1)*Nz, c2 = Nyz + Nz;
	v4sf h, e1, e2, e3, e4;
	v4sf ch = {0.5, 0.5, 0.5, 0.5}; 

	omp_set_num_threads(8);
	#pragma omp parallel for \
	shared( Ntot, Nz, Nyz, c1, c2, ch, Ex, Ey, Ez, Hx, Hy, Hz ) \
   	private( h, e1, e2, e3, e4, idx, i ) \
	schedule( static )
	for ( idx=0; idx<Ntot; idx+=4 ) {
		i = idx + idx/c1*Nz + c2;

		h  = LOAD( &Hx[i] );
		e1 = LOAD( &Ez[i] );
		e2 = LOAD( &Ez[i-Nz] );
		e3 = LOAD( &Ey[i] );
		e4 = LOAD( &Ey[i-1] );
		STORE( &Hx[i], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4)))) ); 

		h  = LOAD( &Hy[i] );
		e1 = LOAD( &Ex[i] );
		e2 = LOAD( &Ex[i-1] );
		e3 = LOAD( &Ez[i] );
		e4 = LOAD( &Ez[i-Nyz] );
		STORE( &Hy[i], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4)))) ); 

		h  = LOAD( &Hz[i] );
		e1 = LOAD( &Ey[i] );
		e2 = LOAD( &Ey[i-Nyz] );
		e3 = LOAD( &Ex[i] );
		e4 = LOAD( &Ex[i-Nz] );
		STORE( &Hz[i], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4)))) ); 
	}

}


int main() {
	int tstep;
	char time_str[32];
	time_t t0;
	int i;

	// --------------------------------------------------------------------------------
	// Set the parameters
	N3 N;
	N.x = 300;
	N.y = 300;
	N.z = 304;
	int TMAX = 1000;
	
	float S = 0.5;
	float dx = 10e-9;
	float dt = S*dx/light_velocity;

	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);

	// --------------------------------------------------------------------------------
	// Allocate host memory
	P3F3 CE;
	CE.x = makeArray3D( N.x, N.y, N.z, 0 );
	CE.y = makeArray3D( N.x, N.y, N.z, 0 );
	CE.z = makeArray3D( N.x, N.y, N.z, 0 );

	P3F3 E;
	E.x = makeArray3D( N.x, N.y, N.z, 0 );
	E.y = makeArray3D( N.x, N.y, N.z, 0 );
	E.z = makeArray3D( N.x, N.y, N.z, 0 );

	P3F3 H;
	H.x = makeArray3D( N.x, N.y, N.z, 0 );
	H.y = makeArray4D( N.x, N.y, N.z, 0 );
	H.z = makeArray3D( N.x, N.y, N.z, 0 );

	// --------------------------------------------------------------------------------
	// Geometry
	set_geometry( N, CE );

	// --------------------------------------------------------------------------------
	// time loop
	int Ntot = (N.x-1)*(N.y-1)*N.z;
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++) {
		updateE ( Ntot, N.y, N.z, E.x[0][0], E.y[0][0], E.z[0][0], H.x[0][0], H.y[0][0], H.z[0][0], CE.x[0][0], CE.y[0][0], CE.z[0][0] );
		
		for ( i=0; i<N.z; i++ ) E.z[N.x/2-10][N.y/2-20][i] += sin(0.1*tstep);

		updateH ( Ntot, N.y, N.z, E.x[0][0], E.y[0][0], E.z[0][0], H.x[0][0], H.y[0][0], H.z[0][0] );

		/*	
		if ( tstep/50*50 == tstep ) {
			dumpToH5(N.x, N.y, N.z, 0, 0, N.z/2, N.x-1, N.y-1, N.z/2, E.z, "png/Ez-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered png/Ez-%05d.h5", tstep);
			
			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
		*/
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);

}
