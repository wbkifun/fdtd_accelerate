#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>

#define Npml 15

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


void set_geometry(N3 N, P3F3 CE) {
	int i,j,k;

	for (i=1; i<N.x-1; i++) {
		for (j=1; j<N.y-1; j++) {
			for (k=1; k<N.z-1; k++) {
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
}


void updateSrc(N3 N, P3F3 E, int tstep) {
	int i;
	//for (i=0; i<N.x-1; i++) E.x[i][N.y/2][N.z/2] += sin(0.1*tstep);
	for (i=0; i<N.z-1; i++) E.z[N.x/2][N.y/2][i] += sin(0.1*tstep);
}


void updateCPMLxE(
		N3 N, P3F3 E, P3F3 H, P3F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float ***psi1, float ***psi2, 
		float *b, float *a,
		int backward) {
	int i,j,k;
	int pi = backward*Npml;

	for (i=is; i<ie; i++) {
		for (j=js; j<je; j++) {
			for (k=ks; k<ke; k++) {
				//printf("pi=%d, [%d, %d, %d]\n", pi, i, j, k);
				psi1[pi][j][k] = b[pi]*psi1[pi][j][k] + a[pi]*(H.z[i+1][j][k] - H.z[i][j][k]);
				E.y[i][j][k] -= CE.y[i][j][k]*psi1[pi][j][k];

				psi2[pi][j][k] = b[pi]*psi2[pi][j][k] + a[pi]*(H.y[i+1][j][k] - H.y[i][j][k]);
				E.z[i][j][k] += CE.z[i][j][k]*psi2[pi][j][k];
			}
		}
		pi++;
	}
}


void updateCPMLxH(
		N3 N, P3F3 E, P3F3 H, P3F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float ***psi1, float ***psi2, 
		float *b, float *a,
		int backward) {
	int i,j,k;
	int pi = backward*Npml;

	for (i=is; i<ie; i++) {
		for (j=js; j<je; j++) {
			for (k=ks; k<ke; k++) {
				psi1[pi][j][k] = b[pi]*psi1[pi][j][k] + a[pi]*(E.z[i][j][k] - E.z[i-1][j][k]);
				H.y[i][j][k] += 0.5*psi1[pi][j][k];

				psi2[pi][j][k] = b[pi]*psi2[pi][j][k] + a[pi]*(E.y[i][j][k] - E.y[i-1][j][k]);
				H.z[i][j][k] -= 0.5*psi2[pi][j][k];
			}
		}
		pi++;
	}
}


void updateCPMLyE(
		N3 N, P3F3 E, P3F3 H, P3F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float ***psi1, float ***psi2, 
		float *b, float *a,
		int backward) {
	int i,j,k;
	int pj;

	for (i=is; i<ie; i++) {
		pj = backward*Npml;
		for (j=js; j<je; j++) {
			for (k=ks; k<ke; k++) {
				//printf("pj=%d, [%d, %d, %d]\n", pj, i, j, k);
				psi1[i][pj][k] = b[pj]*psi1[i][pj][k] + a[pj]*(H.x[i][j+1][k] - H.x[i][j][k]);
				E.z[i][j][k] -= CE.z[i][j][k]*psi1[i][pj][k];

				psi2[i][pj][k] = b[pj]*psi2[i][pj][k] + a[pj]*(H.z[i][j+1][k] - H.z[i][j][k]);
				E.x[i][j][k] += CE.x[i][j][k]*psi2[i][pj][k];
			}
			pj++;
		}
	}
}


void updateCPMLyH(
		N3 N, P3F3 E, P3F3 H, P3F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float ***psi1, float ***psi2, 
		float *b, float *a,
		int backward) {
	int i,j,k;
	int pj;

	for (i=is; i<ie; i++) {
		pj = backward*Npml;
		for (j=js; j<je; j++) {
			for (k=ks; k<ke; k++) {
				psi1[i][pj][k] = b[pj]*psi1[i][pj][k] + a[pj]*(E.x[i][j][k] - E.x[i][j-1][k]);
				H.z[i][j][k] += 0.5*psi1[i][pj][k];

				psi2[i][pj][k] = b[pj]*psi2[i][pj][k] + a[pj]*(E.z[i][j][k] - E.z[i][j-1][k]);
				H.x[i][j][k] -= 0.5*psi2[i][pj][k];
			}
			pj++;
		}
	}
}


void updateCPMLzE(
		N3 N, P3F3 E, P3F3 H, P3F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float ***psi1, float ***psi2, 
		float *b, float *a,
		int backward) {
	int i,j,k;
	int pk;

	for (i=is; i<ie; i++) {
		for (j=js; j<je; j++) {
			pk = backward*Npml;
			for (k=ks; k<ke; k++) {
				//printf("pj=%d, [%d, %d, %d]\n", pj, i, j, k);
				psi1[i][j][pk] = b[pk]*psi1[i][j][pk] + a[pk]*(H.y[i][j][k+1] - H.y[i][j][k]);
				E.x[i][j][k] -= CE.x[i][j][k]*psi1[i][j][pk];

				psi2[i][j][pk] = b[pk]*psi2[i][j][pk] + a[pk]*(H.x[i][j][k+1] - H.x[i][j][k]);
				E.y[i][j][k] += CE.y[i][j][k]*psi2[i][j][pk];

				pk++;
			}
		}
	}
}


void updateCPMLzH(
		N3 N, P3F3 E, P3F3 H, P3F3 CE, 
		int is, int ie, int js, int je, int ks, int ke,
		float ***psi1, float ***psi2, 
		float *b, float *a,
		int backward) {
	int i,j,k;
	int pk;

	for (i=is; i<ie; i++) {
		for (j=js; j<je; j++) {
			pk = backward*Npml;
			for (k=ks; k<ke; k++) {
				psi1[i][j][pk] = b[pk]*psi1[i][j][pk] + a[pk]*(E.y[i][j][k] - E.y[i][j][k-1]);
				H.x[i][j][k] += 0.5*psi1[i][j][pk];

				psi2[i][j][pk] = b[pk]*psi2[i][j][pk] + a[pk]*(E.x[i][j][k] - E.x[i][j][k-1]);
				H.y[i][j][k] -= 0.5*psi2[i][j][pk];

				pk++;
			}
		}
	}
}


int main() {
	int i;
	int tstep;
	char time_str[32];
	time_t t0;

	// Set the parameters
	N3 N;
	N.x = 200;
	N.y = 200;
	N.z = 320;
	int TMAX = 1000;

	float S = 0.5;
	float dx = 10e-9;
	float dt = S*dx/light_velocity;
	printf("N(%d,%d,%d), TMAX=%d\n", N.x, N.y, N.z, TMAX);
	printf("NPML=%d\n", Npml);

	// Allocate host memory
	P3F3 E, H, CE;
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
	set_geometry( N, CE );

/*
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
	}
	free(sigmaE);
	free(sigmaH);

	// Allocate host memory for CPML
	P3F3 psixE, psiyE, psizE;
	P3F3 psixH, psiyH, psizH;

	psixE.y = makeArray3D( 2*Npml, N.y, N.z );
	psixE.z = makeArray3D( 2*Npml, N.y, N.z );

	psiyE.z = makeArray3D( N.x, 2*Npml, N.z );
	psiyE.x = makeArray3D( N.x, 2*Npml, N.z );

	psizE.x = makeArray3D( N.x, N.y, 2*Npml );
	psizE.y = makeArray3D( N.x, N.y, 2*Npml );

	psixH.y = makeArray3D( 2*Npml, N.y, N.z );
	psixH.z = makeArray3D( 2*Npml, N.y, N.z );

	psiyH.z = makeArray3D( N.x, 2*Npml, N.z );
	psiyH.x = makeArray3D( N.x, 2*Npml, N.z );

	psizH.x = makeArray3D( N.x, N.y, 2*Npml );
	psizH.y = makeArray3D( N.x, N.y, 2*Npml );
*/
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++ ) {
		updateE( N, E, H, CE );
		/*
		updateCPMLxE( N, E, H, CE, 1, Npml+1, 1, N.y-1, 1, N.z-1, psixE.y, psixE.z, bE, aE, 0 );
		updateCPMLxE( N, E, H, CE, N.x-(Npml+1), N.x-1, 1, N.y-1, 1, N.z-1, psixE.y, psixE.z, bE, aE, 1 );
		updateCPMLyE( N, E, H, CE, 1, N.x-1, 1, Npml+1, 1, N.z-1, psiyE.z, psiyE.x, bE, aE, 0 );
		updateCPMLyE( N, E, H, CE, 1, N.x-1, N.y-(Npml+1), N.y-1, 1, N.z-1, psiyE.z, psiyE.x, bE, aE, 1 );
		updateCPMLzE( N, E, H, CE, 1, N.x-1, 1, N.y-1, 1, Npml+1, psizE.x, psizE.y, bE, aE, 0 );
		updateCPMLzE( N, E, H, CE, 1, N.x-1, 1, N.y-1, N.z-(Npml+1), N.z-1, psizE.x, psizE.y, bE, aE, 1 );
		*/

		updateSrc( N, E, tstep );

		updateH( N, E, H );
		/*
		updateCPMLxH( N, E, H, CE, 1, Npml+1, 1, N.y, 1, N.z, psixH.y, psixH.z, bH, aH, 0 );
		updateCPMLxH( N, E, H, CE, N.x-Npml, N.x, 1, N.y, 1, N.z, psixH.y, psixH.z, bH, aH, 1 );
		updateCPMLyH( N, E, H, CE, 1, N.x, 1, Npml+1, 1, N.z, psiyH.z, psiyH.x, bH, aH, 0 );
		updateCPMLyH( N, E, H, CE, 1, N.x, N.y-Npml, N.y, 1, N.z, psiyH.z, psiyH.x, bH, aH, 1 );
		updateCPMLzH( N, E, H, CE, 2, N.x, 1, N.y, 1, Npml+1, psizH.x, psizH.y, bH, aH, 0 );
		updateCPMLzH( N, E, H, CE, 1, N.x, 1, N.y, N.z-Npml, N.z, psizH.x, psizH.y, bH, aH, 1 );
		*/
		
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
