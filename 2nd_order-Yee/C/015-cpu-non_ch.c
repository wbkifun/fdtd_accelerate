#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

const float light_velocity = 2.99792458e8;	// m s- 
const float ep0 = 8.85418781762038920e-12;	// F m-1 (permittivity at vacuum)
const float	mu0 = 1.25663706143591730e-6;	// N A-2 (permeability at vacuum)
//const float imp0 = sqrt( mu0/ep0 );	// (impedance at vacuum)
const float pi = 3.14159265358979323846;


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


float ***makeArray3D(int Nx, int Ny, int Nz) {
	float ***f;
	int i;

	f = (float ***) calloc (Nx, sizeof(float **));
	f[0] = (float **) calloc (Ny*Nx, sizeof(float *));
	f[0][0] = (float *) calloc (Nz*Ny*Nx, sizeof(float));

	for (i=0; i<Nx; i++) f[i] = f[0] + i*Ny;
	for (i=0; i<Ny*Nx; i++) f[0][i] = f[0][0] + i*Nz;

	return f;
}


void updateE(N3 N, P3F3 E, P3F3 H, P3F3 CE) {
	int i,j,k;

	for (i=0; i<N.x-1; i++) {
		for (j=0; j<N.y-1; j++) {
			for (k=0; k<N.z-1; k++) {
				E.x[i][j][k] += CE.x[i][j][k]*(H.z[i][j+1][k] - H.z[i][j][k] - H.y[i][j][k+1] + H.y[i][j][k]);
				E.y[i][j][k] += CE.y[i][j][k]*(H.x[i][j][k+1] - H.x[i][j][k] - H.z[i+1][j][k] + H.z[i][j][k]);
				E.z[i][j][k] += CE.z[i][j][k]*(H.y[i+1][j][k] - H.y[i][j][k] - H.x[i][j+1][k] + H.x[i][j][k]);
			}
		}
	}

	i=N.x-1;
	for (j=1; j<N.y-1; j++) for (k=1; k<N.z-1; k++) 
		E.x[i][j][k] += CE.x[i][j][k]*(H.z[i][j+1][k] - H.z[i][j][k] - H.y[i][j][k+1] + H.y[i][j][k]);

	j=N.y-1;
	for (i=1; i<N.x-1; i++) for (k=1; k<N.z-1; k++) 
		E.y[i][j][k] += CE.y[i][j][k]*(H.x[i][j][k+1] - H.x[i][j][k] - H.z[i+1][j][k] + H.z[i][j][k]);

	k=N.z-1;
	for (i=1; i<N.x-1; i++) for (j=1; j<N.y-1; j++) 
		E.z[i][j][k] += CE.z[i][j][k]*(H.y[i+1][j][k] - H.y[i][j][k] - H.x[i][j+1][k] + H.x[i][j][k]);
}


void updateH(N3 N, P3F3 E, P3F3 H) {
	int i,j,k;

	for (i=1; i<N.x; i++) {
		for (j=1; j<N.y; j++) {
			for (k=1; k<N.z; k++) {
				H.x[i][j][k] -= 0.5*(E.z[i][j][k] - E.z[i][j-1][k] - E.y[i][j][k] + E.y[i][j][k-1]);
				H.y[i][j][k] -= 0.5*(E.x[i][j][k] - E.x[i][j][k-1] - E.z[i][j][k] + E.z[i-1][j][k]);
				H.z[i][j][k] -= 0.5*(E.y[i][j][k] - E.y[i-1][j][k] - E.x[i][j][k] + E.x[i][j-1][k]);
			}
		}
	}

	i=0;
	for (j=1; j<N.y; j++) for (k=1; k<N.z; k++)
		H.x[i][j][k] -= 0.5*(E.z[i][j][k] - E.z[i][j-1][k] - E.y[i][j][k] + E.y[i][j][k-1]);

	j=0;
	for (i=1; i<N.x; i++) for (k=1; k<N.z; k++) 
		H.y[i][j][k] -= 0.5*(E.x[i][j][k] - E.x[i][j][k-1] - E.z[i][j][k] + E.z[i-1][j][k]);

	k=0;
	for (i=1; i<N.x; i++) for (j=1; j<N.y; j++) 
		H.z[i][j][k] -= 0.5*(E.y[i][j][k] - E.y[i-1][j][k] - E.x[i][j][k] + E.x[i][j-1][k]);
}


void updateSrc(N3 N, P3F3 E, int tstep) {
	int i;
	//for (i=0; i<N.x-1; i++) E.x[i][N.y/2][N.z/2] += sin(0.1*tstep);
	for (i=0; i<N.z-1; i++) E.z[N.x/2][N.y/2][i] += sin(0.1*tstep);
}


int main() {
	int i,j,k;
	int tstep;
	char time_str[32];
	time_t t0;

	// Set the parameters
	N3 N;
	N.x = 240;
	N.y = 256;
	//N.z = 304;
	N.z = 256;
	int TMAX = 100;

	float S = 0.5;
	float dx = 10e-9;
	float dt = S*dx/light_velocity;
	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);

	// Allocate host memory
	P3F3 E, H, CE, CH;
	E.x = makeArray3D( N.x, N.y, N.z );
	E.y = makeArray3D( N.x, N.y, N.z );
	E.z = makeArray3D( N.x, N.y, N.z );
	H.x = makeArray3D( N.x, N.y, N.z );
	H.y = makeArray3D( N.x, N.y, N.z );
	H.z = makeArray3D( N.x, N.y, N.z );
	CE.x = makeArray3D( N.x, N.y, N.z );
	CE.y = makeArray3D( N.x, N.y, N.z );
	CE.z = makeArray3D( N.x, N.y, N.z );

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
			}
		}
	}

	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++ ) {
		updateE( N, E, H, CE );
		
		updateSrc( N, E, tstep );

		updateH( N, E, H );
/*
		if ( tstep/50*50 == tstep ) {
			dumpToH5(N.x, N.y, N.z, N.x/2, 0, 0, N.x/2, N.y-1, N.z-1, E.x, "cpu_png/Ex-%05d.h5", tstep);
			exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered cpu_png/Ex-%05d.h5", tstep);
			//dumpToH5(N.x, N.y, N.z, 0, 0, N.z/2, N.x-1, N.y-1, N.z/2, E.z, "cpu_png/Ez-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -z0 -S4 -c /usr/share/h5utils/colormaps/dkbluered cpu_png/Ez-%05d.h5", tstep);

			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
*/
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);

}
