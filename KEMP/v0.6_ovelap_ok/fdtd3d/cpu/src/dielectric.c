#include <Python.h>
#include <numpy/arrayobject.h>
#include <xmmintrin.h>
#include <omp.h>
#define OMP_MAX_THREADS 4
#define LOADU _mm_loadu_ps	// not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps

static PyObject *update_h(PyObject *self, PyObject *args) {
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	if (!PyArg_ParseTuple(args, "OOOOOO", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz )) return NULL;

	int nx, ny, nz, idx;
	float *ex, *ey, *ez, *hx, *hy, *hz;
	nx = (int)(Ex->dimensions)[0];
	ny = (int)(Ex->dimensions)[1];
	nz = (int)(Ex->dimensions)[2];
	ex = (float*)(Ex->data);
	ey = (float*)(Ey->data);
	ez = (float*)(Ez->data);
	hx = (float*)(Hx->data);
	hy = (float*)(Hy->data);
	hz = (float*)(Hz->data);

	__m128 ex0, ey0, ez0, e1, e2, h, ch={0.5,0.5,0.5,0.5};
	omp_set_num_threads(OMP_MAX_THREADS);
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
   	Py_INCREF(Py_None);
   	return Py_None;
}

static PyObject *update_e(PyObject *self, PyObject *args) {
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	PyArrayObject *CEx, *CEy, *CEz;
	if (!PyArg_ParseTuple(args, "OOOOOOOOO", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz, &CEx, &CEy, &CEz )) return NULL;

	int nx, ny, nz, idx;
	float *ex, *ey, *ez, *hx, *hy, *hz, *cex, *cey, *cez;
	nx = (int)(Ex->dimensions)[0];
	ny = (int)(Ex->dimensions)[1];
	nz = (int)(Ex->dimensions)[2];
	ex = (float*)(Ex->data);
	ey = (float*)(Ey->data);
	ez = (float*)(Ez->data);
	hx = (float*)(Hx->data);
	hy = (float*)(Hy->data);
	hz = (float*)(Hz->data);
	cex = (float*)(CEx->data);
	cey = (float*)(CEy->data);
	cez = (float*)(CEz->data);

	__m128 hx0, hy0, hz0, h1, h2, e, ce;
	omp_set_num_threads(OMP_MAX_THREADS);
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
   	Py_INCREF(Py_None);
   	return Py_None;
}

/* =============================================================
 * method table listing
 * module's initialization
============================================================= */
static PyMethodDef cfunc_methods[] = {
	{"update_h", update_h, METH_VARARGS, ""},
	{"update_e", update_e, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdielectric() {
	Py_InitModule3("dielectric", cfunc_methods, "");
	import_array();
}
