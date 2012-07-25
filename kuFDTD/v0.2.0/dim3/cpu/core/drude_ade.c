/*
 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 7. 4
 last update  :

 Copyright : GNU GPL
*/

#include <base.h>


static PyObject *update_e(PyObject *self, PyObject *args) {
	int Ncore, Nx, Ny, Nz;
	int idx0, spaceNyz, spaceNz;
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *CEx, *CEy, *CEz;
	PyArrayObject *Jx, *Jy, *Jz;
	PyArrayObject *CJAx, *CJAy, *CJAz;
	PyArrayObject *CJBx, *CJBy, *CJBz;

	if (!PyArg_ParseTuple(args, "iiiiiiiOOOOOOOOOOOOOOO",
				&Ncore, &Nx, &Ny, &Nz, &idx0, &spaceNyz, &spaceNz,
				&Ex, &Ey, &Ez, &CEx, &CEy, &CEz,
			    &Jx, &Jy, &Jz, &CJAx, &CJAy, &CJAz, &CJBx, &CJBy, &CJBz	))
   	{ return NULL; }

	float *ex, *ey, *ez;
	float *cex, *cey, *cez;
	float *jx, *jy, *jz;
	float *cjax, *cjay, *cjaz;
	float *cjbx, *cjby, *cjbz;

	ex = (float*)(Ex->data);
	ey = (float*)(Ey->data);
	ez = (float*)(Ez->data);
	cex = (float*)(CEx->data);
	cey = (float*)(CEy->data);
	cez = (float*)(CEz->data);
	jx = (float*)(Jx->data);
	jy = (float*)(Jy->data);
	jz = (float*)(Jz->data);
	cjax = (float*)(CJAx->data);
	cjay = (float*)(CJAy->data);
	cjaz = (float*)(CJAz->data);
	cjbx = (float*)(CJBx->data);
	cjby = (float*)(CJBy->data);
	cjbz = (float*)(CJBz->data);

	int i, j, k, ji, ei;
	v4sf f, cja, cjb, e, ce, e_tmp; 

	/*
	omp_set_num_threads( Ncore );
	#pragma omp parallel for \
	shared( Nz, Nyz, c1, c2, Ex, Ey, Ez, Hx, Hy, Hz, CEx, CEy, CEz ) \
   	private( e, ce, h1, h2, h3, h4, idx, i ) \
	schedule( static )
	*/
	for ( i=0; i<Nx; i++ ) {
		for ( j=0; j<Ny; j++ ) {
			for ( k=0; k<Nz; k+=4 ) {
				ji = i*Ny*Nz + j*Nz + k;
				ei = i*spaceNyz + j*spaceNz + k + idx0;

				f   = LOAD( jx+ji );
				cja = LOAD( cjax+ji );
				cjb = LOAD( cjbx+ji );
				e   = LOAD( ex+ei );
				ce  = LOAD( cex+ei );
				e_tmp = SUB(e, MUL(ce,f));
				STORE( ex+ei, e_tmp );
				STORE( jx+ji, ADD( MUL(cja,f), MUL(cjb,e_tmp) ) ); 
				/*
				tmp = ADD( MUL(cja,f), MUL(cjb,e) );
				STORE( jx+ji, tmp ); 
				STORE( ex+ei, SUB(e, MUL(ce,tmp)) );
				*/

				f   = LOAD( jy+ji );
				cja = LOAD( cjay+ji );
				cjb = LOAD( cjby+ji );
				e   = LOAD( ey+ei );
				ce  = LOAD( cey+ei );
				e_tmp = SUB(e, MUL(ce,f));
				STORE( ey+ei, e_tmp );
				STORE( jy+ji, ADD( MUL(cja,f), MUL(cjb,e_tmp) ) ); 

				f   = LOAD( jz+ji );
				cja = LOAD( cjaz+ji );
				cjb = LOAD( cjbz+ji );
				e   = LOAD( ez+ei );
				ce  = LOAD( cez+ei );
				e_tmp = SUB(e, MUL(ce,f));
				STORE( ez+ei, e_tmp );
				STORE( jz+ji, ADD( MUL(cja,f), MUL(cjb,e_tmp) ) ); 
			}
		}
	}
   	Py_INCREF(Py_None);
   	return Py_None;
}



/*========================================================================
 * method table listing
 * module's initialization
========================================================================*/
static char update_e_doc[] = "";
static char module_doc[] = "";
static PyMethodDef drude_ade_methods[] = {
	{"update_e", update_e, METH_VARARGS, update_e_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdrude_ade() {
	Py_InitModule3("drude_ade", drude_ade_methods, module_doc);
	import_array();
}
