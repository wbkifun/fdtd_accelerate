#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *update_h(PyObject *self, PyObject *args) {
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	if (!PyArg_ParseTuple(args, "OOOOOO", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz )) return NULL;

	int nx, ny, nz, i, j, k, idx;
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

   	Py_INCREF(Py_None);
   	return Py_None;
}

static PyObject *update_e(PyObject *self, PyObject *args) {
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	PyArrayObject *CEx, *CEy, *CEz;
	if (!PyArg_ParseTuple(args, "OOOOOOOOO", &Ex, &Ey, &Ez, &Hx, &Hy, &Hz, &CEx, &CEy, &CEz )) return NULL;

	int nx, ny, nz, i, j, k, idx;
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
