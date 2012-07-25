#include <omp.h>
#include <xmmintrin.h>
#define LOADU _mm_loadu_ps // not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps
#define SET1 _mm_set_ps1

void update_e(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz , float *cex, float *cey, float *cez) {
	int idx, i, j, k;
	__m128 hx0, hy0, hz0, h1, h2, e, ce, mask_exy={1, 1, 1, 0}, mask_z={1, 1, 1, 1};

	#pragma omp parallel for private(idx, i, j, k, hx0, hy0, hz0, h1, h2, e , ce)
	for( idx=0; idx<nx*ny*nz; idx+=4 ) {
        hx0 = LOAD(hx+idx);
        hy0 = LOAD(hy+idx);
        hz0 = LOAD(hz+idx);

        i = idx/(ny*nz);
        j = (idx - i*ny*nz)/nz;
        k = idx%nz;

        if( j<ny-1 ) {
            e = LOAD(ex+idx);
            ce = LOAD(cex+idx);
            h1 = LOAD(hz+idx+nz);
            h2 = LOADU(hy+idx+1);
            e = ADD(e, MUL(ce ,SUB(SUB(h1, hz0), SUB(h2, hy0))));
            if( k == nz-4 ) e = MUL(e, mask_exy);
            STORE(ex+idx, e);
        }

        if( i<nx-1 ) {
            e = LOAD(ey+idx); 
            ce = LOAD(cey+idx);
            h1 = LOADU(hx+idx+1);
            h2 = LOAD(hz+idx+ny*nz);
            e = ADD(e, MUL(ce, SUB(SUB(h1, hx0), SUB(h2, hz0))));
            if( k == nz-4 ) e = MUL(e, mask_exy);
            STORE(ey+idx, e);
        }

        if( i<nx-1 && j<ny-1 ) {
            e = LOAD(ez+idx);
            ce = LOAD(cez+idx);
            h1 = LOAD(hy+idx+ny*nz);
            h2 = LOAD(hx+idx+nz);
            e = ADD(e, MUL(ce, SUB(SUB(h1, hy0), SUB(h2, hx0))));
            if( k == nz-4 ) e = MUL(e, mask_z);
            STORE(ez+idx, e);
        }
	}
}



void update_h(int nx, int ny, int nz, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz ) {
	int idx, i, j, k;
	__m128 ex0, ey0, ez0, e1, e2, h, ch=SET1(0.5), mask0={0, 1, 1, 1}, mask_z={1, 1, 1, 1};

	#pragma omp parallel for private(idx, i, j, k, ex0, ey0, ez0, e1, e2, h )
	for( idx=0; idx<nx*ny*nz; idx+=4 ) {
        ex0 = LOAD(ex+idx);
        ey0 = LOAD(ey+idx);
        ez0 = LOAD(ez+idx);

        i = idx/(ny*nz);
        j = (idx - i*ny*nz)/nz;
        k = idx%nz;

        if( j>0 ) {
            h = LOAD(hx+idx);
            
            e1 = LOAD(ez+idx-nz);
            e2 = LOADU(ey+idx-1);
            h = SUB(h, MUL(ch, SUB(SUB(ez0, e1), SUB(ey0, e2))));
            if( k == 0 ) h = MUL(h, mask0);
            if( k == nz-4 ) h = MUL(h, mask_z);
            STORE(hx+idx, h);
        }

        if( i>0 ) {
            h = LOAD(hy+idx);
            
            e1 = LOADU(ex+idx-1);
            e2 = LOAD(ez+idx-ny*nz);
            h = SUB(h, MUL(ch, SUB(SUB(ex0, e1), SUB(ez0, e2))));
            if( k == 0 ) h = MUL(h, mask0);
            if( k == nz-4 ) h = MUL(h, mask_z);
            STORE(hy+idx, h);
        }

        if( i>0 && j>0 ) {
            h = LOAD(hz+idx);
            
            e1 = LOAD(ey+idx-ny*nz);
            e2 = LOAD(ex+idx-nz);
            h = SUB(h, MUL(ch, SUB(SUB(ey0, e1), SUB(ex0, e2))));
            if( k == nz-4 ) h = MUL(h, mask_z);
            STORE(hz+idx, h);
        }
	}
}

