#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <xmmintrin.h>
#define LOADU _mm_loadu_ps	// not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps

void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int idx;
	__m128 ex0, ey0, ez0, e1, e2, h, ch={0.5,0.5,0.5,0.5};
	#pragma omp parallel for \
	shared(nx, ny, nz, ex, ey, ez, hx, hy, hz, ch) \
	private(ex0, ey0, ez0, e1, e2, h, idx) \
	schedule(guided)
	for ( idx=nz; idx<nx*ny*nz; idx+=4 ) {
		ex0 = LOAD(ex+idx);
		ey0 = LOAD(ey+idx);
		ez0 = LOAD(ez+idx);

		h = LOAD(hx+idx);
		e1 = LOAD(ez+idx-nz);
		e2 = LOADU(ey+idx-1);
		STORE(hx+idx, SUB(h,MUL(ch,SUB(SUB(ez0,e1),SUB(ey0,e2)))));

		if( idx > ny*nz ) {
			h = LOAD(hy+idx);
			e1 = LOADU(ex+idx-1);
			e2 = LOAD(ez+idx-ny*nz);
			STORE(hy+idx, SUB(h,MUL(ch,SUB(SUB(ex0,e1),SUB(ez0,e2)))));

			h = LOAD(hz+idx);
			e1 = LOADU(ey+idx-ny*nz);
			e2 = LOAD(ex+idx-nz);
			STORE(hz+idx, SUB(h,MUL(ch,SUB(SUB(ey0,e1),SUB(ex0,e2)))));
		}
	}
}

void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx;
	__m128 hx0, hy0, hz0, h1, h2, e, ce;
	#pragma omp parallel for \
	shared(nx, ny, nz, ex, ey, ez, hx, hy, hz, cex, cey, cez) \
	private(hx0, hy0, hz0, h1, h2, e, ce, idx) \
	schedule(guided)
	for ( idx=0; idx<nx*ny*(nz-1); idx+=4 ) {
		hx0 = LOAD(hx+idx);
		hy0 = LOAD(hy+idx);
		hz0 = LOAD(hz+idx);

		e = LOAD(ex+idx);
		ce = LOAD(cex+idx);
		h1 = LOAD(hz+idx+nz);
		h2 = LOADU(hy+idx+1);
		STORE(ex+idx, ADD(e,MUL(ce,SUB(SUB(h1,hz0),SUB(h2,hy0)))));

		if( idx < (nx-1)*ny*nz ) {
			e = LOAD(ey+idx);
			ce = LOAD(cey+idx);
			h1 = LOADU(hx+idx+1);
			h2 = LOAD(hz+idx+ny*nz);
			STORE(ey+idx, ADD(e,MUL(ce,SUB(SUB(h1,hx0),SUB(h2,hz0)))));

			e = LOAD(ez+idx);
			ce = LOAD(cez+idx);
			h1 = LOADU(hy+idx+ny*nz);
			h2 = LOAD(hx+idx+nz);
			STORE(ez+idx, ADD(e,MUL(ce,SUB(SUB(h1,hy0),SUB(h2,hx0)))));
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
