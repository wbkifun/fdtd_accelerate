#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <hdf5.h>


void updateTimer(time_t t0, int tstep, char str[]) {
	int elapsedTime=(int)(time(0)-t0);
	//int estimatedTime=elapsedTime*TMAX/tstep;
	sprintf(str, "%02d:%02d:%02d", elapsedTime/3600, elapsedTime%3600/60, elapsedTime%60);
	/*
	sprintf(str, "%02d:%02d:%02d/%02d:%02d:%02d (%4.1f%%)",
		elapsedTime/3600, elapsedTime%3600/60, elapsedTime%60,
		estimatedTime/3600, estimatedTime%3600/60, estimatedTime%60,
		100.0*(tstep)/TMAX);
	*/
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


float ***makeArray(int Nx, int Ny, int Nz) {
	float ***f;
	int i;

	f = (float ***) calloc (Nx, sizeof(float **));
	f[0] = (float **) calloc (Ny*Nx, sizeof(float *));
	f[0][0] = (float *) calloc (Nz*Ny*Nx, sizeof(float));

	for (i=0; i<Nx; i++) f[i] = f[0] + i*Ny;
	for (i=0; i<Ny*Nx; i++) f[0][i] = f[0][0] + i*Nz;

	return f;
}


void set_geometry(int Nx, int Ny, int Nz, 
		float ***CEx, float ***CEy, float ***CEz) {
	int i,j,k;

	for (i=0; i<Nx-1; i++) {
		for (j=0; j<Ny-1; j++) {
			for (k=0; k<Nz-1; k++) {
				CEx[i][j][k] = 0.5;
				CEy[i][j][k] = 0.5;
				CEz[i][j][k] = 0.5;
			}
		}
	}
}


void updateE(int Nx, int Ny, int Nz,
		float ***Ex, float ***Ey, float ***Ez, 
		float ***Hx, float ***Hy, float ***Hz, 
		float ***CEx, float ***CEy, float ***CEz) {
	int i,j,k;

	for (i=0; i<Nx-1; i++) {
		for (j=0; j<Ny-1; j++) {
			for (k=0; k<Nz-1; k++) {
				Ex[i][j][k] += CEx[i][j][k]*(Hz[i][j+1][k] - Hz[i][j][k] - Hy[i][j][k+1] + Hy[i][j][k]);
				Ey[i][j][k] += CEy[i][j][k]*(Hx[i][j][k+1] - Hx[i][j][k] - Hz[i+1][j][k] + Hz[i][j][k]);
				Ez[i][j][k] += CEz[i][j][k]*(Hy[i+1][j][k] - Hy[i][j][k] - Hx[i][j+1][k] + Hx[i][j][k]);
			}
		}
	}

	i=Nx-1;
	for (j=0; j<Ny-1; j++) for (k=0; k<Nz-1; k++) 
		Ex[i][j][k] += CEx[i][j][k]*(Hz[i][j+1][k] - Hz[i][j][k] - Hy[i][j][k+1] + Hy[i][j][k]);

	j=Ny-1;
	for (i=0; i<Nx-1; i++) for (k=0; k<Nz-1; k++) 
		Ey[i][j][k] += CEy[i][j][k]*(Hx[i][j][k+1] - Hx[i][j][k] - Hz[i+1][j][k] + Hz[i][j][k]);

	k=Nz-1;
	for (i=0; i<Nx-1; i++) for (j=0; j<Ny-1; j++) 
		Ez[i][j][k] += CEz[i][j][k]*(Hy[i+1][j][k] - Hy[i][j][k] - Hx[i][j+1][k] + Hx[i][j][k]);
}


void updateSrc(int Nx, int Ny, int Nz,
		float ***Ex, int tstep) {
	int i;
	for (i=0; i<Nx-1; i++) Ex[i][Ny/2][Nz/2] += sin(0.1*tstep);
}


void updateH( int Nx, int Ny, int Nz,
		float ***Ex, float ***Ey, float ***Ez, 
		float ***Hx, float ***Hy, float ***Hz) {
	int i,j,k;

	for (i=1; i<Nx; i++) {
		for (j=1; j<Ny; j++) {
			for (k=1; k<Nz; k++) {
				Hx[i][j][k] -= 0.5*(Ez[i][j][k] - Ez[i][j-1][k] - Ey[i][j][k] + Ey[i][j][k-1]);
				Hy[i][j][k] -= 0.5*(Ex[i][j][k] - Ex[i][j][k-1] - Ez[i][j][k] + Ez[i-1][j][k]);
				Hz[i][j][k] -= 0.5*(Ey[i][j][k] - Ey[i-1][j][k] - Ex[i][j][k] + Ex[i][j-1][k]);
			}
		}
	}
}


int main() {
	int tstep;
	char time_str[34];
	time_t t0;

	// Set the parameters
	int Nx, Ny, Nz, TMAX;
	Nx = 200;
	Ny = 200;
	Nz = 208;
	TMAX = 100;

	// Allocate host memory
	float ***Ex, ***Ey, ***Ez;
	float ***Hx, ***Hy, ***Hz;
	float ***CEx, ***CEy, ***CEz;
	Ex = makeArray(Nx, Ny, Nz);
	Ey = makeArray(Nx, Ny, Nz);
	Ez = makeArray(Nx, Ny, Nz);
	Hx = makeArray(Nx, Ny, Nz);
	Hy = makeArray(Nx, Ny, Nz);
	Hz = makeArray(Nx, Ny, Nz);
	CEx = makeArray(Nx, Ny, Nz);
	CEy = makeArray(Nx, Ny, Nz);
	CEz = makeArray(Nx, Ny, Nz);

	// Geometry
	set_geometry(Nx, Ny, Nz, CEx, CEy, CEz);

	// Update on the CPU
	t0 = time(0);
	for ( tstep=1; tstep<=TMAX; tstep++) {
		updateE(Nx, Ny, Nz, Ex, Ey, Ez, Hx, Hy, Hz, CEx, CEy, CEz);
		updateSrc(Nx, Ny, Nz, Ex, tstep);
		updateH(Nx, Ny, Nz, Ex, Ey, Ez, Hx, Hy, Hz);
		
		if ( tstep/100*100 == tstep ) {
			//dumpToH5(Nx, Ny, Nz, Nx/2, 0, 0, Nx/2, Ny-1, Nz-1, Ex, "cpu_png/Ex-%05d.h5", tstep);
			//exec("h5topng -ZM0.1 -x0 -S4 -c /usr/share/h5utils/colormaps/dkbluered cpu_png/Ex-%05d.h5", tstep);
			updateTimer(t0, tstep, time_str);
			printf("tstep=%d\t%s\n", tstep, time_str);
		}
	}
	updateTimer(t0, tstep, time_str);
	printf("tstep=%d\t%s\n", tstep, time_str);
}
