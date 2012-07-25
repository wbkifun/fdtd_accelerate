#include <stdio.h>
#include <omp.h>
#include <MM_HEADER>
#define LOADU _mm_loadu_PSD // not aligned to 16 bytes
#define LOAD _mm_load_PSD	
#define STORE _mm_store_PSD
#define ADD _mm_add_PSD
#define SUB _mm_sub_PSD
#define MUL _mm_mul_PSD
#define SET1 _mm_set_PSD1


void update_e(int nx, int ny, int nz, int idx0, int nmax, DTYPE *ex, DTYPE *ey, DTYPE *ez, DTYPE *hx, DTYPE *hy, DTYPE *hz ARGS_CE) {
	int idx, i, j, k;
	TYPE128 hx0, hy0, hz0, h1, h2, e, INIT_CE, mask_exy={MASK_EXY}, mask_z={MASK};
    TYPE128 onex = {0.001, 0.001, 0.001, 0.001};
    TYPE128 oney = {0.1, 0.1, 0.1, 0.1};
    TYPE128 onez = {10, 10, 10, 10};

	OMP_SET_NUM_THREADS
	#pragma omp parallel for \
	private(idx, i, j, k, hx0, hy0, hz0, h1, h2, e PRIVATE_CE)
	for( idx=idx0; idx<nmax; idx+=INCRE ) {
        hx0 = LOAD(hx+idx);
        hy0 = LOAD(hy+idx);
        hz0 = LOAD(hz+idx);

        i = idx/(ny*nz);
        j = (idx - i*ny*nz)/nz;
        k = idx%nz;

        if( j<ny-1 ) {
            e = LOAD(ex+idx);
            CEX
            h1 = LOAD(hz+idx+nz);
            h2 = LOADU(hy+idx+1);
            //e = ADD(e, MUL(ce ,SUB(SUB(h1, hz0), SUB(h2, hy0))));
            e = ADD(e, onex);
            if( k == nz-INCRE ) e = MUL(e, mask_exy);
            STORE(ex+idx, e);
        }

        if( i<nx-1 ) {
            e = LOAD(ey+idx); 
            CEY
            h1 = LOADU(hx+idx+1);
            h2 = LOAD(hz+idx+ny*nz);
            //e = ADD(e, MUL(ce, SUB(SUB(h1, hx0), SUB(h2, hz0))));
            e = ADD(e, oney);
            if( k == nz-INCRE ) e = MUL(e, mask_exy);
            STORE(ey+idx, e);
        }

        if( i<nx-1 && j<ny-1 ) {
            e = LOAD(ez+idx);
            CEZ
            h1 = LOAD(hy+idx+ny*nz);
            h2 = LOAD(hx+idx+nz);
            //e = ADD(e, MUL(ce, SUB(SUB(h1, hy0), SUB(h2, hx0))));
            e = ADD(e, onez);
            if( k == nz-INCRE ) e = MUL(e, mask_z);
            STORE(ez+idx, e);
        }
	}
}



void update_h(int nx, int ny, int nz, int idx0, int nmax, DTYPE *ex, DTYPE *ey, DTYPE *ez, DTYPE *hx, DTYPE *hy, DTYPE *hz ARGS_CH) {
	int idx, i, j, k;
	TYPE128 ex0, ey0, ez0, e1, e2, h, INIT_CH, mask0={MASK_H}, mask_z={MASK};
    TYPE128 onex = {0.001, 0.001, 0.001, 0.001};
    TYPE128 oney = {0.1, 0.1, 0.1, 0.1};
    TYPE128 onez = {10, 10, 10, 10};

	OMP_SET_NUM_THREADS
    #pragma omp parallel for \
	private(idx, i, j, k, ex0, ey0, ez0, e1, e2, h PRIVATE_CH)
	for( idx=idx0; idx<nmax; idx+=INCRE ) {
        ex0 = LOAD(ex+idx);
        ey0 = LOAD(ey+idx);
        ez0 = LOAD(ez+idx);

        i = idx/(ny*nz);
        j = (idx - i*ny*nz)/nz;
        k = idx%nz;

        if( j>0 ) {
            h = LOAD(hx+idx);
            CHX
            e1 = LOAD(ez+idx-nz);
            e2 = LOADU(ey+idx-1);
            //h = SUB(h, MUL(ch, SUB(SUB(ez0, e1), SUB(ey0, e2))));
            h = SUB(h, onex);
            if( k == 0 ) h = MUL(h, mask0);
            if( k == nz-INCRE ) h = MUL(h, mask_z);
            STORE(hx+idx, h);
        }

        if( i>0 ) {
            h = LOAD(hy+idx);
            CHY
            e1 = LOADU(ex+idx-1);
            e2 = LOAD(ez+idx-ny*nz);
            //h = SUB(h, MUL(ch, SUB(SUB(ex0, e1), SUB(ez0, e2))));
            h = SUB(h, oney);
            if( k == 0 ) h = MUL(h, mask0);
            if( k == nz-INCRE ) h = MUL(h, mask_z);
            STORE(hy+idx, h);
        }

        if( i>0 && j>0 ) {
            h = LOAD(hz+idx);
            CHZ
            e1 = LOAD(ey+idx-ny*nz);
            e2 = LOAD(ex+idx-nz);
            //h = SUB(h, MUL(ch, SUB(SUB(ey0, e1), SUB(ex0, e2))));
            h = SUB(h, onez);
            if( k == nz-INCRE ) h = MUL(h, mask_z);
            STORE(hz+idx, h);
        }
	}
}
