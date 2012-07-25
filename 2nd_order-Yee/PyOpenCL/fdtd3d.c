#include <xmmintrin.h>
#include <omp.h>
#define LOADU _mm_loadu_ps	// not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps
#define SET _mm_set_ps1


void update_h(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int idx;
	__m128 ex0, ey0, ez0, e1, e2, h, ch={0.5,0.5,0.5,0.5};

	omp_set_num_threads(OMP_MAX_THREADS);
	#pragma omp parallel for \
	shared(ex, ey, ez, hx, hy, hz, ch) \
	private(ex0, ey0, ez0, e1, e2, h, idx) \
	schedule(guided)
	for ( idx=NZ; idx<NXYZ; idx+=4 ) {
		ex0 = LOAD(ex+idx);
		ey0 = LOAD(ey+idx);
		ez0 = LOAD(ez+idx);

		h = LOAD(hx+idx);
		e1 = LOAD(ez+idx-NZ);
		e2 = LOADU(ey+idx-1);
		STORE(hx+idx, SUB(h,MUL(ch,SUB(SUB(ez0,e1),SUB(ey0,e2)))));

		if( idx > NYZ ) {
			h = LOAD(hy+idx);
			e1 = LOADU(ex+idx-1);
			e2 = LOAD(ez+idx-NYZ);
			STORE(hy+idx, SUB(h,MUL(ch,SUB(SUB(ex0,e1),SUB(ez0,e2)))));

			h = LOAD(hz+idx);
			e1 = LOADU(ey+idx-NYZ);
			e2 = LOAD(ex+idx-NZ);
			STORE(hz+idx, SUB(h,MUL(ch,SUB(SUB(ey0,e1),SUB(ex0,e2)))));
		}
	}
}


void update_e(float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int idx;
	__m128 hx0, hy0, hz0, h1, h2, e, ce;

	omp_set_num_threads(OMP_MAX_THREADS);
	#pragma omp parallel for \
	shared(ex, ey, ez, hx, hy, hz, cex, cey, cez) \
	private(hx0, hy0, hz0, h1, h2, e, ce, idx) \
	schedule(guided)
	for ( idx=0; idx<NXY*(NZ-1); idx+=4 ) {
		hx0 = LOAD(hx+idx);
		hy0 = LOAD(hy+idx);
		hz0 = LOAD(hz+idx);

		e = LOAD(ex+idx);
		ce = LOAD(cex+idx);
		h1 = LOAD(hz+idx+NZ);
		h2 = LOADU(hy+idx+1);
		STORE(ex+idx, ADD(e,MUL(ce,SUB(SUB(h1,hz0),SUB(h2,hy0)))));

		if( idx < (NX-1)*NYZ ) {
			e = LOAD(ey+idx);
			ce = LOAD(cey+idx);
			h1 = LOADU(hx+idx+1);
			h2 = LOAD(hz+idx+NYZ);
			STORE(ey+idx, ADD(e,MUL(ce,SUB(SUB(h1,hx0),SUB(h2,hz0)))));

			e = LOAD(ez+idx);
			ce = LOAD(cez+idx);
			h1 = LOADU(hy+idx+NYZ);
			h2 = LOAD(hx+idx+NZ);
			STORE(ez+idx, ADD(e,MUL(ce,SUB(SUB(h1,hy0),SUB(h2,hx0)))));
		}
	}
}
