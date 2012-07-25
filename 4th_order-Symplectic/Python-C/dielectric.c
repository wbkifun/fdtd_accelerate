#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *update_h(PyObject *self, PyObject *args) {
	int nx, ny, nz;
	float cl;
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	PyArrayObject *Cx, *Cy, *Cz;
	if (!PyArg_ParseTuple(args, "iiifOOOOOOOOO",
				&nx, &ny, &nz, &cl,
				&Ex, &Ey, &Ez, &Hx, &Hy, &Hz, &Cx, &Cy, &Cz ))
   	{ return NULL; }

	float *ex, *ey, *ez;
	float *hx, *hy, *hz;
	float *cx, *cy, *cz;
	ex = (float*)(Ex->data);
	ey = (float*)(Ey->data);
	ez = (float*)(Ez->data);
	hx = (float*)(Hx->data);
	hy = (float*)(Hy->data);
	hz = (float*)(Hz->data);
	cx = (float*)(Cx->data);
	cy = (float*)(Cy->data);
	cz = (float*)(Cz->data);

	int i, j, k, idx;
	for( k=0; k<nz-1; k++) {
		for( j=0; j<ny-1; j++) {
			for( i=0; i<nx-1; i++) {
				idx = i + j*nx + k*nx*ny;
				if( j>1 && j<ny-1 && k>1 && k<nz-1 ) 
					hx[idx] -= cl*cx[idx]*( 27*( ez[idx] - ez[idx-nx] - ey[idx] + ey[idx-nx*ny] )
											- ( ez[idx+nx] - ez[idx-2*nx] - ey[idx+nx*ny] + ey[idx-2*nx*ny] ) );
				if( i>1 && i<nx-1 && k>1 && k<nz-1 ) 
					hy[idx] -= cl*cy[idx]*( 27*( ex[idx] - ex[idx-nx*ny] - ez[idx] + ez[idx-1] )
											- ( ex[idx+nx*ny] - ex[idx-2*nx*ny] - ez[idx+1] + ez[idx-2] ) );
				if( i>1 && i<nx-1 && j>1 && j<ny-1 ) 
					hz[idx] -= cl*cz[idx]*( 27*( ey[idx] - ey[idx-1] - ex[idx] + ex[idx-nx] )
											- ( ey[idx+1] - ey[idx-2] - ex[idx+nx] + ex[idx-2*nx] ) );
			}
		}
	}

   	Py_INCREF(Py_None);
   	return Py_None;
}


static PyObject *update_e(PyObject *self, PyObject *args) {
	int nx, ny, nz;
	float dl;
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	PyArrayObject *Cx, *Cy, *Cz;
	if (!PyArg_ParseTuple(args, "iiifOOOOOOOOO",
				&nx, &ny, &nz, &dl,
				&Ex, &Ey, &Ez, &Hx, &Hy, &Hz, &Cx, &Cy, &Cz ))
   	{ return NULL; }

	float *ex, *ey, *ez;
	float *hx, *hy, *hz;
	float *cx, *cy, *cz;
	ex = (float*)(Ex->data);
	ey = (float*)(Ey->data);
	ez = (float*)(Ez->data);
	hx = (float*)(Hx->data);
	hy = (float*)(Hy->data);
	hz = (float*)(Hz->data);
	cx = (float*)(Cx->data);
	cy = (float*)(Cy->data);
	cz = (float*)(Cz->data);

	int i, j, k, idx;
	for( k=0; k<nz-1; k++) {
		for( j=0; j<ny-1; j++) {
			for( i=0; i<nx-1; i++) {
				idx = i + j*nx + k*nx*ny;
				if( j>0 && j<ny-2 && k>0 && k<nz-2 ) 
					ex[idx] += dl*cx[idx]*( 27*( hz[idx+nx] - hz[idx] - hy[idx+nx*ny] + hy[idx] )
											- ( hz[idx+2*nx] - hz[idx-nx] - hy[idx+2*nx*ny] + hy[idx-nx*ny] ) );
				if( i>0 && i<nx-2 && k>0 && k<nz-2 ) 
					ey[idx] += dl*cy[idx]*( 27*( hx[idx+nx*ny] - hx[idx] - hz[idx+1] + hz[idx] )
											- ( hx[idx+2*nx*ny] - hx[idx-nx*ny] - hz[idx+2] + hz[idx-1] ) );
				if( i>0 && i<nx-2 && j>0 && j<ny-2 ) 
					ez[idx] += dl*cz[idx]*( 27*( hy[idx+1] - hy[idx] - hx[idx+nx] + hx[idx] )
											- ( hy[idx+2] - hy[idx-1] - hx[idx+2*nx] + hx[idx-nx] ) );
			}
		}
	}

   	Py_INCREF(Py_None);
   	return Py_None;
}


// for python module initialization
static PyMethodDef dielectric_methods[] = {
	{"update_h", update_h, METH_VARARGS, ""},
	{"update_e", update_e, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdielectric() {
	Py_InitModule3("dielectric", dielectric_methods, "");
	import_array();
}
