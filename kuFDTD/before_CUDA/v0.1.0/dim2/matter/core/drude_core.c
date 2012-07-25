/*
 <File Description>

 File Name : drude_core.c

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 2. 1. Fri

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the core update function for E or H fields for Drude-type media.

===============================================================================
*/

#include <../../../kufdtd_core_base.h>

static PyObject
*update_drude(PyObject *self, PyObject *args)
{
	//=========================================================================
	// import the python object
	//=========================================================================
   	char *info_fieldface;
	char *in_out;
	float dt;
	PyObject *number_cells, *update_field, *base_field, *update_f_field;
	PyObject *gamma_up, *ca, *cb, *cf;

	if (!PyArg_ParseTuple(args, "ssfOOOOOOOO",
		   	&info_fieldface,
		   	&in_out,
		   	&dt,
		   	&number_cells,
		   	&update_field,
		   	&base_field,
		   	&update_f_field,
		   	&gamma_up,
		   	&ca,
		   	&cb,
		   	&cf))
   	{
	   	return NULL;
	}

	//=========================================================================
	// error exception
	//=========================================================================
   	if (strcmp(in_out, IN_FIELD) && strcmp(in_out, OUT_FIELD))
   	{
	   	PyErr_Format(PyExc_ValueError,
			   	"In and Out option is one of two strings. \"in_field\" or \"out_field\".");
	   	return NULL;
   	}
   	if (strcmp(info_fieldface, EFACED) && strcmp(info_fieldface, HFACED))
   	{
	   	PyErr_Format(PyExc_ValueError,
			   	"The information of field face is one of two strings. \"efaced\" or \"hfaced\".");
	   	return NULL;
   	}
   	if (!PyList_Check(update_field) && !PyList_Check(base_field)
		  	 	&& !PyList_Check(update_f_field) && !PyList_Check(gamma_up)
			   	&& !PyList_Check(ca) &&!PyList_Check(cb)
			   	&& !PyList_Check(cf))
   	{
	   	PyErr_Format(PyExc_ValueError,
			   	"Some of arguments are not List, except, info_fieldface, in_out, dt.");
	   	return NULL;
   	}
   	if (!PyList_Check(number_cells) && !PyTuple_Check(number_cells))
   	{
	   	PyErr_Format(PyExc_ValueError,
			   	"number_cells must be List or Tuple.");
	   	return NULL;
   	}
   	else if PyList_Check(number_cells)
   	{
	   	number_cells = PyList_AsTuple(number_cells);
   	}

	//=========================================================================
	// convert the variables fro python object to c
	//=========================================================================
	int N[3];
	PyArrayObject *upx, *upy, *upz;
	PyArrayObject *basex, *basey, *basez;
	PyArrayObject *fupx, *fupy, *fupz;
	PyArrayObject *gamma_upx, *gamma_upy, *gamma_upz;
	PyArrayObject *cax, *cay, *caz;
	PyArrayObject *cbx, *cby, *cbz;
	PyArrayObject *cfx, *cfy, *cfz;

	N[X_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, X_AXIS));
	N[Y_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Y_AXIS));
	N[Z_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Z_AXIS));
	upx = (PyArrayObject *) PyList_GetItem(update_field, X_AXIS);
	upy = (PyArrayObject *) PyList_GetItem(update_field, Y_AXIS);
	upz = (PyArrayObject *) PyList_GetItem(update_field, Z_AXIS);
	basex = (PyArrayObject *) PyList_GetItem(base_field, X_AXIS);
	basey = (PyArrayObject *) PyList_GetItem(base_field, Y_AXIS);
	basez = (PyArrayObject *) PyList_GetItem(base_field, Z_AXIS);
	fupx = (PyArrayObject *) PyList_GetItem(update_f_field, X_AXIS);
	fupy = (PyArrayObject *) PyList_GetItem(update_f_field, Y_AXIS);
	fupz = (PyArrayObject *) PyList_GetItem(update_f_field, Z_AXIS);
	gamma_upx = (PyArrayObject *) PyList_GetItem(gamma_up, X_AXIS);
	gamma_upy = (PyArrayObject *) PyList_GetItem(gamma_up, Y_AXIS);
	gamma_upz = (PyArrayObject *) PyList_GetItem(gamma_up, Z_AXIS);
	cax = (PyArrayObject *) PyList_GetItem(ca, X_AXIS);
	cay = (PyArrayObject *) PyList_GetItem(ca, Y_AXIS);
	caz = (PyArrayObject *) PyList_GetItem(ca, Z_AXIS);
	cbx = (PyArrayObject *) PyList_GetItem(cb, X_AXIS);
	cby = (PyArrayObject *) PyList_GetItem(cb, Y_AXIS);
	cbz = (PyArrayObject *) PyList_GetItem(cb, Z_AXIS);
	cfx = (PyArrayObject *) PyList_GetItem(cf, X_AXIS);
	cfy = (PyArrayObject *) PyList_GetItem(cf, Y_AXIS);
	cfz = (PyArrayObject *) PyList_GetItem(cf, Z_AXIS);

	//=========================================================================
	// set variable coefficients
	//=========================================================================
	int i, j, k;
	int i_start, i_end = N[X_AXIS] + 1;
	int j_start, j_end = N[Y_AXIS] + 1;
	int k_start, k_end = N[Z_AXIS] + 1;
	short c1;  // shifted one cell(index)
	if (!strcmp(in_out, IN_FIELD))
	{
	   	i_start = 0;
	   	j_start = 0;
	   	k_start = 0;
	   	c1 = 1;
	}
	else if (!strcmp(in_out, OUT_FIELD))
	{
	   	i_start = 1;
	   	j_start = 1;
	   	k_start = 1;
	   	c1 = -1;
	}

	short info_EorHfaced;
	if (!strcmp(info_fieldface, HFACED))
	{
	   	info_EorHfaced = 1;
	}
	else
	{
	   	info_EorHfaced = -1;
	}

	float curl_bzy, curl_byz;  // curl of base fields
	float curl_bxz, curl_bzx; 
	float curl_byx, curl_bxy;
	int i_c1, j_c1, k_c1;

	//=========================================================================
	// main loop
	//=========================================================================
   	for (i = i_start; i < i_end; i++)
   	{
	   	for (j = j_start; j < j_end; j++)
	   	{
		   	for (k = k_start; k < k_end; k++)
		   	{
			   	i_c1 = i + c1;
			   	j_c1 = j + c1;
			   	k_c1 = k + c1;

			   	curl_bzy = P3F(basez, i_c1, j_c1, k_c1)
				   		- P3F(basez, i_c1, j   , k_c1);
			   	curl_byz = P3F(basey, i_c1, j_c1, k_c1)
				   		- P3F(basey, i_c1, j_c1, k   );
			   	curl_bxz = P3F(basex, i_c1, j_c1, k_c1)
				   		- P3F(basex, i_c1, j_c1, k   );
			   	curl_bzx = P3F(basez, i_c1, j_c1, k_c1)
				   		- P3F(basez, i   , j_c1, k_c1);
			   	curl_byx = P3F(basey, i_c1, j_c1, k_c1)
				  	 	- P3F(basey, i   , j_c1, k_c1);
			   	curl_bxy = P3F(basex, i_c1, j_c1, k_c1)
				   		- P3F(basex, i_c1, j   , k_c1);

			   	P3F(fupx, i, j, k) = 
						dt * P3F(upx, i, j, k)
					   	+ exp(-dt * P3F(gamma_upx, i, j, k))
					   	* P3F(fupx, i, j, k);
			   	P3F(fupy, i, j, k) =
				  	 	dt * P3F(upy, i, j, k)
					   	+ exp (-dt * P3F(gamma_upy, i, j, k))
					   	* P3F(fupy, i, j, k);
			   	P3F(fupz, i, j, k) =
				   		dt * P3F(upz, i, j, k)
					   	+ exp(-dt * P3F(gamma_upz, i, j, k))
					   	* P3F(fupz, i, j, k);

			   	P3F(upx, i, j, k) =
				   		2. * P3F(cax, i, j, k)
					   	* P3F(upx, i, j, k)
					   	+ info_EorHfaced * P3F(cbx, i, j, k)
					   	* (curl_bzy - curl_byz)
					   	- P3F(cfx, i, j, k) * P3F(fupx, i, j, k);
			   	P3F(upy, i, j, k) =
				   		2. * P3F(cay, i, j, k)
					   	* P3F(upy, i, j, k)
					   	+ info_EorHfaced * P3F(cby, i, j, k)
					   	* (curl_bxz - curl_bzx)
					   	- P3F(cfy, i, j, k) * P3F(fupy, i, j, k);
			   	P3F(upz, i, j, k) =
				   		2. * P3F(caz, i, j, k)
					   	* P3F(upz, i, j, k)
					   	+ info_EorHfaced * P3F(cbz, i, j, k)
					   	* (curl_byx - curl_bxy)
					   	- P3F(cfz, i, j, k) * P3F(fupz, i, j, k);
		   	}
	   	}
   	}

	//=========================================================================
	// secondary loop for bisymmetry
	//=========================================================================
   	if (!strcmp(in_out, OUT_FIELD))
   	{
	   	i = i_end;
	   	i_c1 = i + c1;
	   	for (j = j_start; j < j_end; j++)
	   	{
		   	for (k = k_start; k < k_end; k++)
		   	{
			   	j_c1 = j + c1;
			   	k_c1 = k + c1;

			   	curl_bzy = P3F(basez, i_c1, j_c1, k_c1)
				   		- P3F(basez, i_c1, j   , k_c1);
			   	curl_byz = P3F(basey, i_c1, j_c1, k_c1)
				   		- P3F(basey, i_c1, j_c1, k   );

			   	P3F(fupx, i, j, k) =
				   		dt * P3F(upx, i, j, k)
					   	+ exp(-dt * P3F(gamma_upx, i, j, k))
					   	* P3F(fupx, i, j, k);
			   	P3F(upx, i, j, k) =
				   		2. * P3F(cax, i, j, k)
					   	* P3F(upx, i, j, k)
					   	+ info_EorHfaced * P3F(cbx, i, j, k)
					   	* (curl_bzy - curl_byz)
					   	- P3F(cfx, i, j, k) * P3F(fupx, i, j, k);
		   	}
	   	}

	   	j = j_end;
	   	j_c1 = j + c1;
	   	for (i = i_start; i < i_end; i++)
	   	{
		   	for (k = k_start; k < k_end; k++)
		   	{
			   	i_c1 = i + c1;
			   	k_c1 = k + c1;

			   	curl_bxz = P3F(basex, i_c1, j_c1, k_c1)
				   		- P3F(basex, i_c1, j_c1, k   );
			   	curl_bzx = P3F(basez, i_c1, j_c1, k_c1)
				   		- P3F(basez, i   , j_c1, k_c1);

			   	P3F(fupy, i, j, k) =
					   	dt * P3F(upy, i, j, k)
					   	+ exp(-dt * P3F(gamma_upy, i, j, k))
					   	* P3F(fupy, i, j, k);
			   	P3F(upy, i, j, k) =
				   		2. * P3F(cay, i, j, k)
					   	* P3F(upy, i, j, k)
					   	+ info_EorHfaced * P3F(cby, i, j, k)
					   	* (curl_bxz - curl_bzx)
					   	- P3F(cfy, i, j, k) * P3F(fupy, i, j, k);
		   	}
	   	}

	   	k = k_end;
	   	k_c1 = k + c1;
	   	for (i = i_start; i < i_end; i++)
	   	{
		   	for (j = j_start; j < j_end; j++)
		   	{
			   	i_c1 = i + c1;
			   	j_c1 = j + c1;

			   	curl_byx = P3F(basey, i_c1, j_c1, k_c1)
				  	 	- P3F(basey, i   , j_c1, k_c1);
			   	curl_bxy = P3F(basex, i_c1, j_c1, k_c1)
				   		- P3F(basex, i_c1, j   , k_c1);

			   	P3F(fupz, i, j, k) =
				  	 	dt * P3F(upz, i, j, k)
					   	+ exp(-dt * P3F(gamma_upz, i, j, k))
					   	* P3F(fupz, i, j, k);
			   	P3F(upz, i, j, k) =
				   		2. * P3F(caz, i, j, k)
					   	* P3F(upz, i, j, k)
					   	+ info_EorHfaced * P3F(cbz, i, j, k)
					   	* (curl_byx - curl_bxy)
					   	- P3F(cfz, i, j, k) * P3F(fupz, i, j, k);
		   	}
	   	}
   	}
   
	Py_INCREF(Py_None); 
	return Py_None;
}


/*
===============================================================================
 * method table listing
 * module's initialization
===============================================================================
*/
static char update_drude_doc[] = "update_drude(grid_opt, in_out_field, dt, number_cells, update_field, base_field, update_f_field, gamma_up, ca, cb, cf)";
static char module_doc[] = \
	"module drude_core:\n\
	update_drude(grid_opt, in_out_field, dt, number_cells, update_field, base_field, update_f_field, gamma_up, ca, cb, cf)";

static PyMethodDef drude_core_methods[] = {
	{"update_drude", update_drude, METH_VARARGS, update_drude_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdrude_core() {
	Py_InitModule3("drude_core", drude_core_methods, module_doc);
	import_array();
}
