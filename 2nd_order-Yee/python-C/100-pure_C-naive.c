#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int i, j, k, idx;
	for ( i=0; i<nx; i++ ) {
		for ( j=0; j<ny; j++ ) {
			for ( k=0; k<nz; k++ ) {
				idx = i*ny*nz + j*nz + k;
				if( j>0 && k>0 ) hx[idx] -= 0.5*( ez[idx] - ez[idx-nz] - ey[idx] + ey[idx-1] );
				if( i>0 && k>0 ) hy[idx] -= 0.5*( ex[idx] - ex[idx-1] - ez[idx] + ez[idx-ny*nz] );
				if( i>0 && j>0 ) hz[idx] -= 0.5*( ey[idx] - ey[idx-ny*nz] - ex[idx] + ex[idx-nz] );
			}
		}
	}
}

void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int i, j, k, idx;
	for ( i=0; i<nx; i++ ) {
		for ( j=0; j<ny; j++ ) {
			for ( k=0; k<nz; k++ ) {
				idx = i*ny*nz + j*nz + k;
				if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( hz[idx+nz] - hz[idx] - hy[idx+1] + hy[idx] );
				if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( hx[idx+1] - hx[idx] - hz[idx+ny*nz] + hz[idx] );
				if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+ny*nz] - hy[idx] - hx[idx+nz] + hx[idx] );
			}
		}
	}
}


int main() {
	int i, tn, nx=240, ny=256, nz=256;
	int tmax=100, tgap=10;

	float *ex, *ey, *ez;
	float *hx, *hy, *hz;
	float *cex, *cey, *cez;

	ex = (float *) calloc (nx*ny*nz, sizeof(float));
	ey = (float *) calloc (nx*ny*nz, sizeof(float));
	ez = (float *) calloc (nx*ny*nz, sizeof(float));
	hx = (float *) calloc (nx*ny*nz, sizeof(float));
	hy = (float *) calloc (nx*ny*nz, sizeof(float));
	hz = (float *) calloc (nx*ny*nz, sizeof(float));
	cex = (float *) malloc (nx*ny*nz*sizeof(float));
	cey = (float *) malloc (nx*ny*nz*sizeof(float));
	cez = (float *) malloc (nx*ny*nz*sizeof(float));

	for( i=0; i<nx*ny*nz; i++ ) {
		cex[i] = 0.5;
		cey[i] = 0.5;
		cez[i] = 0.5;
	}

	int flop = nx*ny*nz*30;
	float *flops;
	flops = (float *) calloc (tmax/tgap+1, sizeof(float));
	struct timeval t1, t2;
	float dt;
	gettimeofday(&t1, NULL);

	for( tn=1; tn<=tmax; tn++ ) {
		update_h(nx, ny, nz, ex, ey, ez, hx, hy, hz);
		update_e(nx, ny, nz, ex, ey, ez, hx, hy, hz, cex, cey, cez);
		for( i=0; i<nz; i++ ) ez[(nx/2)*ny*nz+(ny/2)*nz+i] += sin(0.1*tn);

		if( tn%tgap == 0 ) {
			gettimeofday(&t2, NULL);
			dt = (t2.tv_sec + t2.tv_usec*1e-6) - (t1.tv_sec + t1.tv_usec*1e-6);
			flops[tn/tgap] = flop/dt*tgap*1e-9;
			printf("[%1.6f] %d/%d %1.3f GFLOPS\n", dt, tn, tmax, flops[tn/tgap]);
			//fflush();
			gettimeofday(&t1, NULL);
		}
	}

	float flops_avg=0;
	for(i=2; i<tmax/tgap-1; i++ ) flops_avg += flops[i];
	printf("\navg: %1.2f GFLOPS\n", flops_avg/(tmax/tgap-3));

	return 0;
}
