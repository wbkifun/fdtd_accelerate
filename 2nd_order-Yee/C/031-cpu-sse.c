#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

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


void updateE(N3 N, P3F3 E, P3F3 H, P3F3 CE) {
	int i,j,k;
	v4sf e, ce, h1, h2, h3, h4; 

	for (i=0;i<N.x-1;i++){
		for (j=0;j<N.y-1;j++){
			for (k=0;k<N.z;k+=4){
				e = LOAD(&E.x[i][j][k]);
				ce = LOAD(&CE.x[i][j][k]);
				h1 = LOAD(&H.z[i][j+1][k]);
				h2 = LOAD(&H.z[i][j][k]);
				h3 = LOAD(&H.y[i][j][k+1]);
				h4 = LOAD(&H.y[i][j][k]);
				STORE(&E.x[i][j][k], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4))))); 
			}
		}
	}

	for (i=0;i<N.x-1;i++){
		for (j=0;j<N.y-1;j++){
			for (k=0;k<N.z;k+=4){

				e = LOAD(&E.y[i][j][k]);
				ce = LOAD(&CE.y[i][j][k]);
				h1 = LOAD(&H.x[i][j][k+1]);
				h2 = LOAD(&H.x[i][j][k]);
				h3 = LOAD(&H.z[i+1][j][k]);
				h4 = LOAD(&H.z[i][j][k]);
				STORE(&E.y[i][j][k], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4))))); 
			}
		}
	}

	for (i=0;i<N.x-1;i++){
		for (j=0;j<N.y-1;j++){
			for (k=0;k<N.z;k+=4){

				e = LOAD(&E.z[i][j][k]);
				ce = LOAD(&CE.z[i][j][k]);
				h1 = LOAD(&H.y[i+1][j][k]);
				h2 = LOAD(&H.y[i][j][k]);
				h3 = LOAD(&H.x[i][j+1][k]);
				h4 = LOAD(&H.x[i][j][k]);
				STORE(&E.z[i][j][k], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4))))); 
			}
		}
	}

	i=N.x-1;
	for (j=0;j<N.y-1;j++){
		for (k=0;k<N.z;k+=4){
			e = LOAD(&E.x[i][j][k]);
			ce = LOAD(&CE.x[i][j][k]);
			h1 = LOAD(&H.z[i][j+1][k]);
			h2 = LOAD(&H.z[i][j][k]);
			h3 = LOAD(&H.y[i][j][k+1]);
			h4 = LOAD(&H.y[i][j][k]);
			STORE(&E.x[i][j][k], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4))))); 
		}
	}

	j=N.y-1;
	for (i=0;i<N.x-1;i++){
		for (k=0;k<N.z;k+=4){
			e = LOAD(&E.y[i][j][k]);
			ce = LOAD(&CE.y[i][j][k]);
			h1 = LOAD(&H.x[i][j][k+1]);
			h2 = LOAD(&H.x[i][j][k]);
			h3 = LOAD(&H.z[i+1][j][k]);
			h4 = LOAD(&H.z[i][j][k]);
			STORE(&E.y[i][j][k], ADD(e, MUL(ce, SUB( SUB(h1,h2), SUB(h3,h4))))); 
		}
	}
}


void updateH(N3 N, P3F3 E, P3F3 H, P3F3 CH) {
	int i,j,k;
	v4sf h, ch, e1, e2, e3, e4; 

	//omp_set_num_threads(8);
	for (i=1;i<N.x;i++){
		for (j=1;j<N.y;j++){
			for (k=0;k<N.z;k+=4){
				h = LOAD(&H.x[i][j][k]);
				ch = LOAD(&CH.x[i][j][k]);
				e1 = LOAD(&E.z[i][j][k]);
				e2 = LOAD(&E.z[i][j-1][k]);
				e3 = LOAD(&E.y[i][j][k]);
				e4 = LOAD(&E.y[i][j][k-1]);
				STORE(&H.x[i][j][k], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4))))); 
			}
		}
	}

	for (i=1;i<N.x;i++){
		for (j=1;j<N.y;j++){
			for (k=0;k<N.z;k+=4){

				h = LOAD(&H.y[i][j][k]);
				ch = LOAD(&CH.y[i][j][k]);
				e1 = LOAD(&E.x[i][j][k]);
				e2 = LOAD(&E.x[i][j][k-1]);
				e3 = LOAD(&E.z[i][j][k]);
				e4 = LOAD(&E.z[i-1][j][k]);
				STORE(&H.y[i][j][k], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4))))); 
			}
		}
	}
	for (i=1;i<N.x;i++){
		for (j=1;j<N.y;j++){
			for (k=0;k<N.z;k+=4){

				h = LOAD(&H.z[i][j][k]);
				ch = LOAD(&CH.z[i][j][k]);
				e1 = LOAD(&E.y[i][j][k]);
				e2 = LOAD(&E.y[i-1][j][k]);
				e3 = LOAD(&E.x[i][j][k]);
				e4 = LOAD(&E.x[i][j-1][k]);
				STORE(&H.z[i][j][k], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4))))); 
			}
		}
	}

	i=0;
	for (j=1;j<N.y;j++){
		for (k=0;k<N.z;k+=4){
			h = LOAD(&H.x[i][j][k]);
			ch = LOAD(&CH.x[i][j][k]);
			e1 = LOAD(&E.z[i][j][k]);
			e2 = LOAD(&E.z[i][j-1][k]);
			e3 = LOAD(&E.y[i][j][k]);
			e4 = LOAD(&E.y[i][j][k-1]);
			STORE(&H.x[i][j][k], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4))))); 
		}
	}

	j=0;
	for (i=1;i<N.x;i++){
		for (k=0;k<N.z;k+=4){
			h = LOAD(&H.y[i][j][k]);	
			ch = LOAD(&CH.y[i][j][k]);
			e1 = LOAD(&E.x[i][j][k]);
			e2 = LOAD(&E.x[i][j][k-1]);
			e3 = LOAD(&E.z[i][j][k]);
			e4 = LOAD(&E.z[i-1][j][k]);
			STORE(&H.y[i][j][k], SUB(h, MUL(ch, SUB( SUB(e1,e2), SUB(e3,e4))))); 
		}
	}
}


int main() {
	int tstep;
	char time_str[32];
	time_t t0;
	int i,j,k;

	// --------------------------------------------------------------------------------
	// Set the parameters
	N3 N;
	N.x = 300;
	N.y = 300;
	//N.z = 304;
	N.z = 224;
	int TMAX = 100;
	
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

	P3F3 CH;
	CH.x = makeArray3D( N.x, N.y, N.z,0 );
	CH.y = makeArray3D( N.x, N.y, N.z,0 );
	CH.z = makeArray3D( N.x, N.y, N.z,0 );

	P3F3 E;
	E.x = makeArray3D( N.x, N.y, N.z, 0 );
	E.y = makeArray3D( N.x, N.y, N.z, 0 );
	E.z = makeArray3D( N.x, N.y, N.z, 0 );

	P3F3 H;
	H.x = makeArray3D( N.x, N.y, N.z, 0 );
	H.y = makeArray3D( N.x, N.y, N.z, 1 );
	H.z = makeArray3D( N.x, N.y, N.z, 0 );

	// --------------------------------------------------------------------------------
	// Geometry
	for (i=0; i<N.x; i++) {
		for (j=0; j<N.y; j++) {
			for (k=0; k<N.z; k++) {
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

				CH.x[i][j][k] = 0.5;
				CH.y[i][j][k] = 0.5;
				CH.z[i][j][k] = 0.5;

				if ( i == 0 ) {
					CH.y[i][j][k] = 0;
					CH.z[i][j][k] = 0;
				}
				if ( j == 0 ) {
					CH.z[i][j][k] = 0;
					CH.x[i][j][k] = 0;
				}
				if ( k == 0 ) {
					CH.x[i][j][k] = 0;
					CH.y[i][j][k] = 0;
				}
			}
		}
	}

	// --------------------------------------------------------------------------------
	// time loop
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++) {
		updateE ( N, E, H, CE );
		
		for ( i=0; i<N.z; i++ ) E.z[N.x/2-10][N.y/2-20][i] += sin(0.1*tstep);

		updateH ( N, E, H, CH );
		
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
