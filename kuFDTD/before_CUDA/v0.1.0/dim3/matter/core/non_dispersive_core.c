/*
 <File Description>

 File Name : matter_core.c

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 2. 1. Fri

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the core update function for E or H fields for non-dirspersive media

===============================================================================
*/

#include <../../../kufdtd_core_base.h>

static PyObject
*update_non_dispersive(PyObject *self, PyObject *args)
{
	//=========================================================================
	// import the python object
	//=========================================================================
   	char *info_fieldface;
   	char *in_out;
   	PyObject *number_cells, *update_field, *base_field, *cb;

	if (!PyArg_ParseTuple(args, "ssOOOO",
			&info_fieldface,
		   	&in_out,
		   	&number_cells,
		   	&update_field,
		   	&base_field,
		   	&cb))
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
   	if (!PyList_Check(update_field) && !PyList_Check(base_field) && 
			!PyList_Check(cb))
   	{
		PyErr_Format(PyExc_ValueError,
			   	"Some of arguments are not List, except, info_fieldface and in_out.");
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
   	PyArrayObject *cbx, *cby, *cbz;

   	N[X_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, X_AXIS));
   	N[Y_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Y_AXIS));
   	N[Z_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Z_AXIS));
   	upx = (PyArrayObject *) PyList_GetItem(update_field, X_AXIS);
   	upy = (PyArrayObject *) PyList_GetItem(update_field, Y_AXIS);
   	upz = (PyArrayObject *) PyList_GetItem(update_field, Z_AXIS);
   	basex = (PyArrayObject *) PyList_GetItem(base_field, X_AXIS);
   	basey = (PyArrayObject *) PyList_GetItem(base_field, Y_AXIS);
   	basez = (PyArrayObject *) PyList_GetItem(base_field, Z_AXIS);
   	cbx = (PyArrayObject *) PyList_GetItem(cb, X_AXIS);
   	cby = (PyArrayObject *) PyList_GetItem(cb, Y_AXIS); 
	cbz = (PyArrayObject *) PyList_GetItem(cb, Z_AXIS);

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

	int cb_EorH = 0;
   	for (i = 0; i < 3; i++)
   	{
	   	cb_EorH += cbx->dimensions[i];
	   	cb_EorH += cby->dimensions[i];
	   	cb_EorH += cbz->dimensions[i];
   	}
   	int c_EH = 0;  // Conditoin of E field or H field
   	if (cb_EorH <= 9)
   	{
	   	c_EH = 0;
   	}
   	else
   	{
	   	c_EH = 1;
   	}
   	float curl_bzy, curl_byz;  // curl of base fields
   	float curl_bxz, curl_bzx;
   	float curl_byx, curl_bxy;
   	int i_c1, j_c1, k_c1; 
	int c_EH_i, c_EH_j, c_EH_k;

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
			   	c_EH_i = i * c_EH;
			   	c_EH_j = j * c_EH;
			   	c_EH_k = k * c_EH;

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

			   	P3F(upx, i, j, k) += 
						P3F(cbx, c_EH_i, c_EH_j, c_EH_k)
					   	* info_EorHfaced
					   	* (curl_bzy - curl_byz);
			   	P3F(upy, i, j, k) +=
				   		P3F(cby, c_EH_i, c_EH_j, c_EH_k)
					   	* info_EorHfaced
					   	* (curl_bxz - curl_bzx);
			   	P3F(upz, i, j, k) +=
				   		P3F(cbz, c_EH_i, c_EH_j, c_EH_k)
					   	* info_EorHfaced
					   	* (curl_byx - curl_bxy);
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
		c_EH_i = i * c_EH;
	   	for (j = j_start; j < j_end; j++)
	   	{
		   	for (k = k_start; k < k_end; k++)
		   	{
			   	j_c1 = j + c1;
			   	k_c1 = k + c1;
			   	c_EH_j = j * c_EH;
			   	c_EH_k = k * c_EH;

			   	curl_bzy = P3F(basez, i_c1, j_c1, k_c1)
				   		- P3F(basez, i_c1, j   , k_c1);
			   	curl_byz = P3F(basey, i_c1, j_c1, k_c1)
				   		- P3F(basey, i_c1, j_c1, k   );

			   	P3F(upx, i, j, k) += P3F(cbx, c_EH_i, c_EH_j, c_EH_k)
				   		* info_EorHfaced
					   	* (curl_bzy - curl_byz);
		   	}
	   	}

	   	j = j_end;
	   	j_c1 = j + c1;
	   	c_EH_j = j * c_EH;
	   	for (i = i_start; i < i_end; i++)
	   	{
		   	for (k = k_start; k < k_end; k++)
		   	{
			   	i_c1 = i + c1;
			   	k_c1 = k + c1;
			   	c_EH_i = i * c_EH;
			   	c_EH_k = k * c_EH;

			   	curl_bxz = P3F(basex, i_c1, j_c1, k_c1)
				  	 	- P3F(basex, i_c1, j_c1, k   );
			   	curl_bzx = P3F(basez, i_c1, j_c1, k_c1)
				   		- P3F(basez, i   , j_c1, k_c1);

			   	P3F(upy, i, j, k) +=
				   		P3F(cby, c_EH_i, c_EH_j, c_EH_k)
					   	* info_EorHfaced
					   	* (curl_bxz - curl_bzx);
		   	}
	   	}
	   
		k = k_end;
	   	k_c1 = k + c1;
	   	c_EH_k = k * c_EH;
	   	for (i = i_start; i < i_end; i++)
	   	{
		   	for (j = j_start; j < j_end; j++)
		   	{
			   	i_c1 = i + c1; 
				j_c1 = j + c1;
			   	c_EH_i = i * c_EH;
			   	c_EH_j = j * c_EH;

			   	curl_byx = P3F(basey, i_c1, j_c1, k_c1)
				   		- P3F(basey, i   , j_c1, k_c1);
			   	curl_bxy = P3F(basex, i_c1, j_c1, k_c1)
				  	 	- P3F(basex, i_c1, j   , k_c1);

			   	P3F(upz, i, j, k) += P3F(cbz, c_EH_i, c_EH_j, c_EH_k)
				   		* info_EorHfaced
					   	* (curl_byx - curl_bxy);
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
static char update_non_dispersive_doc[] = "update_non_dispersive(grid_opt, in_out_field, number_cells, update_field, base_field, cb)";
static char module_doc[] = \
	"module non_dispersive_core:\n\
	update_non_dispersive(grid_opt, in_out_field, number_cells, update_field, base_field, cb)";

static PyMethodDef non_dispersive_core_methods[] = {
	{"update_non_dispersive", update_non_dispersive, METH_VARARGS, update_non_dispersive_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initnon_dispersive_core() {
	Py_InitModule3("non_dispersive_core", non_dispersive_core_methods, module_doc);
	import_array();
}
