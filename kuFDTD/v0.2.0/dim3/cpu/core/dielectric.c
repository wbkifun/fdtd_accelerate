/*
 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2009. 6. 18
 last update  : 2009. 7. 24

 Copyright : GNU GPL
*/

#include <base.h>
#include <stdlib.h>
#include <pthread.h>

struct thread_args {
	int idx_start, idx_end;
	int Ny, Nz;
	float *ex, *ey, *ez;
	float *hx, *hy, *hz;
	float *cex, *cey, *cez;
};


void *func_update_e( void *args ) {
	struct thread_args *data;
	data = (struct thread_args *)args;

	int idx_start, idx_end;
	int Ny, Nz;
	
	float *ex, *ey, *ez;
	float *hx, *hy, *hz;
	float *cex, *cey, *cez;

	idx_start = data->idx_start;
	idx_end = data->idx_end;
	Ny = data->Ny;
	Nz = data->Nz;
	ex = data->ex;
	ey = data->ey;
	ez = data->ez;
	hx = data->hx;
	hy = data->hy;
	hz = data->hz;

	cex = data->cex;
	cey = data->cey;
	cez = data->cez;

	int idx, i;
	int Nyz = Ny*Nz;
	int c1 = (Ny-1)*Nz, c2 = Nyz + Nz;
	v4sf e, ce, h1, h2, h3, h4, h5; 

	for ( idx=idx_start; idx<idx_end; idx+=4 ) {
		i = idx + idx/c1*Nz + c2;

		h1 = LOAD( hx+i );
		h2 = LOAD( hy+i );
		h3 = LOAD( hz+i );

		e  = LOAD( ex+i );
		ce = LOAD( cex+i );
		h4 = LOAD( hz+i+Nz );
		h5 = LOAD( hy+i+1 );
		STORE( ex+i, ADD(e, MUL(ce, SUB( SUB(h4,h3), SUB(h5,h2)))) ); 

		e  = LOAD( ey+i );
		ce = LOAD( cey+i );
		h4 = LOAD( hx+i+1 );
		h5 = LOAD( hz+i+Nyz );
		STORE( ey+i, ADD(e, MUL(ce, SUB( SUB(h4,h1), SUB(h5,h3)))) ); 

		e  = LOAD( ez+i );
		ce = LOAD( cez+i );
		h4 = LOAD( hy+i+Nyz );
		h5 = LOAD( hx+i+Nz );
		STORE( ez+i, ADD(e, MUL(ce, SUB( SUB(h4,h2), SUB(h5,h1)))) ); 
	}
}


static PyObject *update_e(PyObject *self, PyObject *args) {
	int Ncore, Nx, Ny, Nz;
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;
	PyArrayObject *CEx, *CEy, *CEz;

	if (!PyArg_ParseTuple(args, "iiiiOOOOOOOOO",
				&Ncore, &Nx, &Ny, &Nz,
				&Ex, &Ey, &Ez, &Hx, &Hy, &Hz, &CEx, &CEy, &CEz ))
   	{ return NULL; }

	int Ntot = (Nx-1)*(Ny-1)*Nz;
	int Q = ( Ntot/4 )/Ncore;	// Quotient
	int R = ( Ntot/4 )%Ncore;	// Residual
	int Nthread = Ncore;
	struct thread_args args_list[Nthread];

	pthread_t threads[Nthread];
	pthread_attr_t attr;
	pthread_attr_init( &attr );
	pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );
	pthread_attr_setscope( &attr, PTHREAD_SCOPE_PROCESS );
	int t, rc;
	for ( t=0; t<Nthread; t++ ) {
		args_list[t].idx_start = t*Q*4;
		if ( t < Nthread-1 ) args_list[t].idx_end = (t+1)*Q*4;
		else args_list[t].idx_end = ( (t+1)*Q + R )*4;
		args_list[t].Ny = Ny;
		args_list[t].Nz = Nz;
		args_list[t].ex = (float*)(Ex->data);
		args_list[t].ey = (float*)(Ey->data);
		args_list[t].ez = (float*)(Ez->data);
		args_list[t].hx = (float*)(Hx->data);
		args_list[t].hy = (float*)(Hy->data);
		args_list[t].hz = (float*)(Hz->data);
		args_list[t].cex = (float*)(CEx->data);
		args_list[t].cey = (float*)(CEy->data);
		args_list[t].cez = (float*)(CEz->data);
		rc = pthread_create( &threads[t], &attr, func_update_e, (void *)&args_list[t] );
		if (rc) {
			printf( "Error: return code from pthread_create() is %d\n", rc );
			exit(-1);
		}
	}

	pthread_attr_destroy( &attr );
	void *status;
	for ( t=0; t<Nthread; t++ ) {
		rc = pthread_join( threads[t], &status );
		if (rc) {
			printf( "Error: return code from pthread_join() is %d\n", rc );
			exit(-1);
		}
	}	
	//pthread_exit(NULL);

   	Py_INCREF(Py_None);
   	return Py_None;
}


static PyObject *update_h(PyObject *self, PyObject *args) {
	int Ncore, Nx, Ny, Nz;
	PyArrayObject *Ex, *Ey, *Ez;
	PyArrayObject *Hx, *Hy, *Hz;

	if (!PyArg_ParseTuple(args, "iiiiOOOOOO",
				&Ncore, &Nx, &Ny, &Nz,
				&Ex, &Ey, &Ez, &Hx, &Hy, &Hz ))
   	{ return NULL; }

	float *ex, *ey, *ez;
	float *hx, *hy, *hz;

	ex = (float*)(Ex->data);
	ey = (float*)(Ey->data);
	ez = (float*)(Ez->data);
	hx = (float*)(Hx->data);
	hy = (float*)(Hy->data);
	hz = (float*)(Hz->data);

	int idx, i;
	int Nyz = Ny*Nz;
	int c1 = (Ny-1)*Nz, c2 = Nyz + Nz;
	v4sf h, e1, e2, e3, e4, e5; 
	v4sf ch = {0.5, 0.5, 0.5, 0.5};

	/*
	omp_set_num_threads( Ncore );
	#pragma omp parallel for \
	shared( Nz, Nyz, c1, c2, ch, Ex, Ey, Ez, Hx, Hy, Hz ) \
   	private( h, e1, e2, e3, e4, idx, i ) \
	schedule( static )
	*/
	for ( idx=0; idx<(Nx-1)*(Ny-1)*Nz; idx+=4 ) {
		i = idx + idx/c1*Nz + c2;

		e1 = LOAD( ex+i );
		e2 = LOAD( ey+i );
		e3 = LOAD( ez+i );

		h  = LOAD( hx+i );
		e4 = LOAD( ez+i-Nz );
		e5 = LOAD( ey+i-1 );
		STORE( hx+i, SUB(h, MUL(ch, SUB( SUB(e3,e4), SUB(e2,e5)))) ); 

		h  = LOAD( hy+i );
		e4 = LOAD( ex+i-1 );
		e5 = LOAD( ez+i-Nyz );
		STORE( hy+i, SUB(h, MUL(ch, SUB( SUB(e1,e4), SUB(e3,e5)))) ); 

		h  = LOAD( hz+i );
		e4 = LOAD( ey+i-Nyz );
		e5 = LOAD( ex+i-Nz );
		STORE( hz+i, SUB(h, MUL(ch, SUB( SUB(e2,e4), SUB(e1,e5)))) ); 
	}
   	Py_INCREF(Py_None);
   	return Py_None;
}


/*============================================================================
 * method table listing
 * module's initialization
============================================================================*/
static char update_e_doc[] = "update_e( Ncore, Ny, Nz, Ex, Ey, Ez, Hx, Hy, Hz, CEx, CEy, CEz )";
static char update_h_doc[] = "updatei_h( Ncore, Ny, Nz, Ex, Ey, Ez, Hx, Hy, Hz )";
static char module_doc[] = "module dielectric:\n\
	update_e( Ncore, Ny, Nz, Ex, Ey, Ez, Hx, Hy, Hz, CEx, CEy, CEz )\n\
	update_h( Ncore, Ny, Nz, Ex, Ey, Ez, Hx, Hy, Hz )";

static PyMethodDef dielectric_methods[] = {
	{"update_e", update_e, METH_VARARGS, update_e_doc},
	{"update_h", update_h, METH_VARARGS, update_h_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdielectric() {
	Py_InitModule3("dielectric", dielectric_methods, module_doc);
	import_array();
}
