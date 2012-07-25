/*
 <File Description>

 File Name : cpml_core.c

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 6. 24

 Copyright : GNU LGPL

============================== < File Description > ===========================

Define the update function for CPML.

===============================================================================
*/

#include <../../kufdtd_core_base.h>

static PyObject
*update_cpml_3d(PyObject *self, PyObject *args)
{
	//=========================================================================
	// import the python object
	//=========================================================================
	char *in_out;
	PyObject *number_cells, *update_field, *base_field, *cb;
	float ds;
	int npml;
	PyObject *pml_coefficient, *psi;
	int pml_direction;
	char *pml_position;
	if (!PyArg_ParseTuple(args, "sOOOOfiOOis",
			&in_out,
		   	&number_cells,
		   	&update_field,
		   	&base_field,
		   	&cb,
		   	&ds,
		   	&npml,
		   	&pml_coefficient,
		   	&psi,
		   	&pml_direction,
		   	&pml_position))
	{
	   	return NULL;
	}
	//printf("in core: direction = %d, position = %s\n", pml_direction, pml_position);

	//=========================================================================
	// error exception 
	//=========================================================================
	if (strcmp(in_out, IN_FIELD) && strcmp(in_out, OUT_FIELD))
	{
			PyErr_Format(PyExc_ValueError,
						 "In and Out option is one of two strings. in_field is \"in_field\" and out_field is \"out_field\".");
			return NULL;
	}
	if (pml_direction > 2)
	{
			PyErr_Format(PyExc_ValueError,
						 "The information of direction is one of three integers. Zero is the x direction. One is the y direction. Two is the z direction.");
			return NULL;
	}
	if (strcmp(pml_position, FRONT) && strcmp(pml_position, BACK))
	{
			PyErr_Format(PyExc_ValueError,
						 "PML Position option is one of two position. \"front\" or \"back\".");
			return NULL;
	}
	if (!PyList_Check(update_field) && !PyList_Check(base_field) && !PyList_Check(cb) &&
			!PyList_Check(pml_coefficient) && !PyList_Check(psi))
	{
			PyErr_Format(PyExc_ValueError,
						 "Some of arguments are not List.");
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
	// convert the variables from python object to c
	//=========================================================================
	int N[3];
	int permutation1, permutation2;
	PyArrayObject *up1, *up2;
	PyArrayObject *base1, *base2;
	PyArrayObject *psi_up1, *psi_up2;
	PyArrayObject *kapa, *rcm_b, *mrcm_a;
	PyArrayObject *cb1, *cb2;

	N[X_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, X_AXIS));
	N[Y_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Y_AXIS));
	N[Z_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Z_AXIS));
	permutation1 = (pml_direction + 1) % 3;
	permutation2 = (pml_direction + 2) % 3;
	up1 = (PyArrayObject *) PyList_GetItem(update_field, permutation1);
	up2 = (PyArrayObject *) PyList_GetItem(update_field, permutation2);
	base1 = (PyArrayObject *) PyList_GetItem(base_field, permutation2);
	base2 = (PyArrayObject *) PyList_GetItem(base_field, permutation1);
	psi_up1 = (PyArrayObject *) PyList_GetItem(psi, pml_direction * 2);
	psi_up2 = (PyArrayObject *) PyList_GetItem(psi, pml_direction * 2 + 1);
	kapa = (PyArrayObject *) PyList_GetItem(pml_coefficient, 0);
	rcm_b = (PyArrayObject *) PyList_GetItem(pml_coefficient, 1);
	mrcm_a = (PyArrayObject *) PyList_GetItem(pml_coefficient,2);
	cb1 = (PyArrayObject *) PyList_GetItem(cb, permutation1);
	cb2 = (PyArrayObject *) PyList_GetItem(cb, permutation2);

	//=========================================================================
	// set variable coefficients
	//=========================================================================
	int i, j, k;
	int con[3] = {0, 0, 0};  //condition : con_i, con_j, con_k
	con[pml_direction] = 1;
	int INorOUT = 1;
	int ijkpml = 0;
	int c1[3] = {1, 1, 1};  // ci1, cj1, ck1
	int c2[3] = {1, 1, 1};  // ci2, cj2, ck2
	int start[3] = {0, 0, 0};  // i_start, j_start, k_start
	int end[3] = {N[X_AXIS] + 1, N[Y_AXIS] + 1, N[Z_AXIS] + 1};  // i_end, j_end, k_end
	int pml[3] = {0, 0, 0};  // ipml, jpml, kpml

	if (!strcmp(in_out, IN_FIELD))
	{
	   	if (!strcmp(pml_position, FRONT))
	   	{
		   	end[pml_direction] = npml;
	   	}
	   	else
	   	{
		   	ijkpml = N[pml_direction] + 1 - 2* npml;
		   	start[pml_direction] = ijkpml + npml;
		   	pml[pml_direction] = ijkpml;
	   	}
   	}
   	else
   	{
	   	INorOUT *= -1;
	   	for (i = 0; i < 3; i++)
	   	{
		   	c1[i] *= -1;
		   	c2[i] *= -1;
		   	start[i] += 1;
		   	ijkpml = 1;
	   	}
	   	if (!strcmp(pml_position, FRONT))
	   	{
		   	end[pml_direction] = npml + 1;
		   	pml[pml_direction] = ijkpml;
	   	}
	   	else
	   	{
		   	ijkpml = N[pml_direction] + 1 - 2* npml;
		   	start[pml_direction] = ijkpml + npml;
		   	pml[pml_direction] = ijkpml;
	   	}
   	}

	c2[pml_direction] = 0;
	int cb_EorH = 0;
	for (i = 0; i < 3; i++)
	{
	   	cb_EorH += cb1->dimensions[i];
	   	cb_EorH += cb2->dimensions[i];
   	}
		
   	int c_EH = 0;  // Conditoin of E field or H field
   	int EorH;
   	if (cb_EorH <= 6)	// required modification for conformal PEC
   	{
	   	c_EH = 0;
	   	EorH = -1;
   	}
   	else
   	{
	   	c_EH = 1;
	   	EorH = 1;
   	}

	float curl_b1, curl_b2;  // curl of base fields
	int i_ci1, i_ci2, i_ipml;
	int j_cj1, j_cj2, j_jpml;
	int k_ck1, k_ck2, k_kpml;
	int con_ijkpml;
	int c_EH_i, c_EH_j, c_EH_k;

	/*
	printf("in_out= %s, INorOUT=%d\n", in_out, INorOUT);
	printf("direction= %d, position= %s\n", pml_direction, pml_position);
	printf("permutation1= %d,%d\n", permutation1, permutation2);
	printf("cb_EorH=%d, c_EH=%d, EorH=%d\n", cb_EorH, c_EH, EorH);
	printf("start=%d,%d,%d, end=%d,%d,%d\n",start[0],start[1],start[2],end[0],end[1],end[2]);
	printf("c1=%d,%d,%d, c2=%d,%d,%d\n", c1[0],c1[1],c1[2],c2[0],c2[1],c2[2]);
	printf("pml=%d,%d,%d, ijkpml=%d\n", pml[0],pml[1],pml[2],ijkpml);
	printf("con=%d,%d,%d\n\n", con[0],con[1],con[2]);
	*/

	//=========================================================================
	// main loop
	//=========================================================================
	for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
	{
	   	for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
	   	{
		   	for (k = start[Z_AXIS]; k < end[Z_AXIS]; k++)
		   	{
			   	i_ci1 = i + c1[X_AXIS];
			   	i_ci2 = i + c2[X_AXIS];
			   	j_cj1 = j + c1[Y_AXIS];
			   	j_cj2 = j + c2[Y_AXIS];
			   	k_ck1 = k + c1[Z_AXIS];
			   	k_ck2 = k + c2[Z_AXIS];
			   	i_ipml = i - pml[X_AXIS];
			   	j_jpml = j - pml[Y_AXIS];
			   	k_kpml = k - pml[Z_AXIS];
			   	con_ijkpml = con[X_AXIS]*i
				   		+ con[Y_AXIS]*j
					   	+ con[Z_AXIS]*k
					   	- ijkpml;
			   	c_EH_i = i * c_EH;
			   	c_EH_j = j * c_EH;
			   	c_EH_k = k * c_EH;
			   	curl_b1 = P3F(base1, i_ci1, j_cj1, k_ck1)
					   	- P3F(base1, i_ci2, j_cj2, k_ck2);
			   	curl_b2 = P3F(base2, i_ci1, j_cj1, k_ck1)
				   		- P3F(base2, i_ci2, j_cj2, k_ck2);
			   	//printf("i_ipml=%d, j_jpml=%d, k_kpml=%d\n", i_ipml, j_jpml, k_kpml);
			   	P3F(psi_up1, i_ipml, j_jpml, k_kpml) = 
						P1F(rcm_b, con_ijkpml)
					   	* P3F(psi_up1, i_ipml, j_jpml, k_kpml)
					   	+ INorOUT
					   			* P1F(mrcm_a, con_ijkpml)
					   			* curl_b1;
			   	P3F(psi_up2, i_ipml, j_jpml, k_kpml) = 
						P1F(rcm_b, con_ijkpml)
					   	* P3F(psi_up2, i_ipml, j_jpml, k_kpml)
					   	+ INorOUT
					   			* P1F(mrcm_a, con_ijkpml)
					   			* curl_b2;
			   	P3F(up1, i, j, k) += 
						- EorH
					   	* P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
					   	* (INorOUT
							   	* (1. / P1F(kapa, con_ijkpml) - 1)
							   	* curl_b1
							   	+ ds*P3F(psi_up1, i_ipml, j_jpml, k_kpml));
			   	P3F(up2, i, j, k) +=
				  	 	+ EorH
					   	* P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
					   	* (INorOUT
							   	* (1. / P1F(kapa, con_ijkpml) - 1)
							   	* curl_b2
							   	+ ds*P3F(psi_up2, i_ipml, j_jpml, k_kpml));
		   	}
	   	}
   	}

	//=========================================================================
	// secondary loop for bisymmetry
	//=========================================================================
	if (!strcmp(in_out, OUT_FIELD))
	{
	   	//printf("pml_direction= %d\n",pml_direction);
	   	switch (pml_direction)
	   	{
		   	case X_AXIS:
			   	//printf("Axis= %d\n",X_AXIS);
			   	j = end[Y_AXIS];
			   	j_cj1 = j + c1[Y_AXIS];
			   	j_cj2 = j + c2[Y_AXIS];
			   	c_EH_j = j * c_EH;
			   	for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
			   	{
				   	for (k = start[Z_AXIS]; k < end[Z_AXIS]; k++)
				   	{
					   	i_ci1 = i + c1[X_AXIS];
					   	i_ci2 = i + c2[X_AXIS];
					   	k_ck1 = k + c1[Z_AXIS];
					   	k_ck2 = k + c2[Z_AXIS];
					   	i_ipml = i - pml[X_AXIS];
					   	k_kpml = k - pml[Z_AXIS];
					   	con_ijkpml = con[X_AXIS]*i
						  	 	+ con[Y_AXIS]*j
						   		+ con[Z_AXIS]*k
							   	- ijkpml;
					   	c_EH_i = i * c_EH;
					   	c_EH_k = k * c_EH;
					   	curl_b1 = P3F(base1, i_ci1, j_cj1, k_ck1)
						   		- P3F(base1, i_ci2, j_cj2, k_ck2);
					   	P3F(psi_up1, i_ipml, 0, k_kpml) = 
								P1F(rcm_b, con_ijkpml)
							   	* P3F(psi_up1, i_ipml, 0, k_kpml)
							   	+ INorOUT
							   	* P1F(mrcm_a, con_ijkpml)
							   	* curl_b1;
					   	P3F(up1, i, j, k) += 
								- EorH
							   	* P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
							   	* (INorOUT
									   	* (1. / P1F(kapa, con_ijkpml) - 1)
									   	* curl_b1
									   	+ ds*P3F(psi_up1, i_ipml, 0, k_kpml));
				   	}
			   	}
			   	k = end[Z_AXIS];
			   	
				k_ck1 = k + c1[Z_AXIS]; k_ck2 = k + c2[Z_AXIS];
			   	c_EH_k = k * c_EH;
			   	for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
			   	{
				   	for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
				   	{
					   	i_ci1 = i + c1[X_AXIS];
					   	i_ci2 = i + c2[X_AXIS];
					   	j_cj1 = j + c1[Y_AXIS];
					   	j_cj2 = j + c2[Y_AXIS];
					   	i_ipml = i - pml[X_AXIS];
					   	j_jpml = j - pml[Y_AXIS];
					   	con_ijkpml = con[X_AXIS]*i
						  	 	+ con[Y_AXIS]*j
						   		+ con[Z_AXIS]*k
							   	- ijkpml;
					   	c_EH_i = i * c_EH;
					   	c_EH_j = j * c_EH;
					   	curl_b2 = P3F(base2, i_ci1, j_cj1, k_ck1)
						   		- P3F(base2, i_ci2, j_cj2, k_ck2);
					   	P3F(psi_up2, i_ipml, j_jpml, 0) = 
								P1F(rcm_b, con_ijkpml)
						   		* P3F(psi_up2, i_ipml, j_jpml, 0)
							   	+ INorOUT
							   	* P1F(mrcm_a, con_ijkpml)
							   	* curl_b2;
					   	P3F(up2, i, j, k) +=
						   	+ EorH
						   	* P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
						   	* (INorOUT
								   	* (1. / P1F(kapa, con_ijkpml) - 1)
													* curl_b2
													+ ds*P3F(psi_up2, i_ipml, j_jpml, 0));
					}
				}
				break;

			 case Y_AXIS:
				//printf("Axis= %d\n",Y_AXIS);
				k = end[Z_AXIS];
				k_ck1 = k + c1[Z_AXIS];
				k_ck2 = k + c2[Z_AXIS];
				c_EH_k = k * c_EH;
				for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
				{
					for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
					{
						i_ci1 = i + c1[X_AXIS];
						i_ci2 = i + c2[X_AXIS];
						j_cj1 = j + c1[Y_AXIS];
						j_cj2 = j + c2[Y_AXIS]; 
						i_ipml = i - pml[X_AXIS];
						j_jpml = j - pml[Y_AXIS];
						con_ijkpml = con[X_AXIS]*i
								+ con[Y_AXIS]*j
								+ con[Z_AXIS]*k
								- ijkpml;
						c_EH_i = i * c_EH;
						c_EH_j = j * c_EH;
						curl_b1 = P3F(base1, i_ci1, j_cj1, k_ck1)
								- P3F(base1, i_ci2, j_cj2, k_ck2);
						P3F(psi_up1, i_ipml, j_jpml, 0) = 
								P1F(rcm_b, con_ijkpml)
								* P3F(psi_up1, i_ipml, j_jpml, 0)
								+ INorOUT
								* P1F(mrcm_a, con_ijkpml)
								* curl_b1;
						P3F(up1, i, j, k) += 
							- EorH
							* P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
							* (INorOUT
									* (1. / P1F(kapa, con_ijkpml) - 1)
									* curl_b1
									+ ds*P3F(psi_up1, i_ipml, j_jpml, 0));
					}
				}
			   
				i = end[X_AXIS];
				i_ci1 = i + c1[X_AXIS];
				i_ci2 = i + c2[X_AXIS];
				c_EH_i = i * c_EH;
				for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
				{
					for (k = start[Z_AXIS]; k < end[Z_AXIS]; k++)
					{
						j_cj1 = j + c1[Y_AXIS];
						j_cj2 = j + c2[Y_AXIS];
						k_ck1 = k + c1[Z_AXIS];
						k_ck2 = k + c2[Z_AXIS];
						j_jpml = j - pml[Y_AXIS];
						k_kpml = k - pml[Z_AXIS];
						con_ijkpml = con[X_AXIS]*i
								+ con[Y_AXIS]*j
								+ con[Z_AXIS]*k
								- ijkpml;
						c_EH_j = j * c_EH;
						c_EH_k = k * c_EH;
						curl_b2 = P3F(base2, i_ci1, j_cj1, k_ck1)
								- P3F(base2, i_ci2, j_cj2, k_ck2);
						P3F(psi_up2, 0, j_jpml, k_kpml) = 
								P1F(rcm_b, con_ijkpml)
								* P3F(psi_up2, 0, j_jpml, k_kpml)
								+ INorOUT
								* P1F(mrcm_a, con_ijkpml)
								* curl_b2;
						P3F(up2, i, j, k) +=
								+ EorH
								* P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
								* (INorOUT
										* (1. / P1F(kapa, con_ijkpml) - 1)
										* curl_b2
										+ ds*P3F(psi_up2, 0, j_jpml, k_kpml));
					}
				}
				break;

			 case Z_AXIS:
				//printf("Axis= %d\n",Z_AXIS);
				i = end[X_AXIS];
				i_ci1 = i + c1[X_AXIS];
				i_ci2 = i + c2[X_AXIS];
				c_EH_i = i * c_EH;
				for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
				{
					for (k = start[Z_AXIS]; k < end[Z_AXIS]; k++)
					{
						j_cj1 = j + c1[Y_AXIS];
						j_cj2 = j + c2[Y_AXIS];
						k_ck1 = k + c1[Z_AXIS];
						k_ck2 = k + c2[Z_AXIS];
						j_jpml = j - pml[Y_AXIS];
						k_kpml = k - pml[Z_AXIS];
						con_ijkpml = con[X_AXIS]*i
								+ con[Y_AXIS]*j
								+ con[Z_AXIS]*k
								- ijkpml;
						c_EH_j = j * c_EH;
						c_EH_k = k * c_EH;
						curl_b1 = P3F(base1, i_ci1, j_cj1, k_ck1)
								- P3F(base1, i_ci2, j_cj2, k_ck2);
						P3F(psi_up1, 0, j_jpml, k_kpml) = 
								P1F(rcm_b, con_ijkpml)
								* P3F(psi_up1, 0, j_jpml, k_kpml)
								+ INorOUT
								* P1F(mrcm_a, con_ijkpml)
								* curl_b1;
						P3F(up1, i, j, k) += 
								- EorH
								* P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
								* (INorOUT
										* (1. / P1F(kapa, con_ijkpml) - 1)
										* curl_b1
										+ ds*P3F(psi_up1, 0, j_jpml, k_kpml));
					}
				}
				j = end[Y_AXIS];
				j_cj1 = j + c1[Y_AXIS];
				j_cj2 = j + c2[Y_AXIS];
				c_EH_j = j * c_EH;
				for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
				{ 
					for (k = start[Z_AXIS]; k < end[Z_AXIS]; k++)
					{
						i_ci1 = i + c1[X_AXIS];
						i_ci2 = i + c2[X_AXIS];
						k_ck1 = k + c1[Z_AXIS];
						k_ck2 = k + c2[Z_AXIS];
						i_ipml = i - pml[X_AXIS]; 
						k_kpml = k - pml[Z_AXIS]; 
						con_ijkpml = con[X_AXIS]*i
								+ con[Y_AXIS]*j
								+ con[Z_AXIS]*k
								- ijkpml;
						c_EH_i = i * c_EH;
						c_EH_k = k * c_EH;
						curl_b2 = P3F(base2, i_ci1, j_cj1, k_ck1)
								- P3F(base2, i_ci2, j_cj2, k_ck2);
						P3F(psi_up2, i_ipml, 0, k_kpml) = 
								P1F(rcm_b, con_ijkpml)
								* P3F(psi_up2, i_ipml, 0, k_kpml)
								+ INorOUT
								* P1F(mrcm_a, con_ijkpml)
								* curl_b2;
						P3F(up2, i, j, k) +=
								+ EorH
								* P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
								* (INorOUT
										* (1. / P1F(kapa, con_ijkpml) - 1)
										* curl_b2
										+ ds*P3F(psi_up2, i_ipml, 0, k_kpml));
					}
				}
				break;
	   	}
   	}

   	Py_INCREF(Py_None);
   	return Py_None;
}



static PyObject
*update_cpml_2update_1base(PyObject *self, PyObject *args)
{
	//=========================================================================
	// import the python object
	//=========================================================================
	char *in_out;
	PyObject *number_cells, *update_field, *base_field, *cb;
	float ds;
	int npml;
	PyObject *pml_coefficient, *psi;
	int pml_direction;
	char *pml_position;
	if (!PyArg_ParseTuple(args, "sOOOOfiOOis",
			&in_out,
		   	&number_cells,
		   	&update_field,
		   	&base_field,
		   	&cb,
		   	&ds,
		   	&npml,
		   	&pml_coefficient,
		   	&psi,
		   	&pml_direction,
		   	&pml_position))
	{
	   	return NULL;
	}
	//printf("in core: direction = %d, position = %s\n", pml_direction, pml_position);

	//=========================================================================
	// error exception 
	//=========================================================================
	if (strcmp(in_out, IN_FIELD) && strcmp(in_out, OUT_FIELD))
	{
			PyErr_Format(PyExc_ValueError,
						 "In and Out option is one of two strings. in_field is \"in_field\" and out_field is \"out_field\".");
			return NULL;
	}
	if (pml_direction > 2)
	{
			PyErr_Format(PyExc_ValueError,
						 "The information of direction is one of three integers. Zero is the x direction. One is the y direction. Two is the z direction.");
			return NULL;
	}
	if (strcmp(pml_position, FRONT) && strcmp(pml_position, BACK))
	{
			PyErr_Format(PyExc_ValueError,
						 "PML Position option is one of two position. \"front\" or \"back\".");
			return NULL;
	}
	if (!PyList_Check(update_field) && !PyList_Check(base_field) && !PyList_Check(cb) &&
			!PyList_Check(pml_coefficient) && !PyList_Check(psi))
	{
			PyErr_Format(PyExc_ValueError,
						 "Some of arguments are not List.");
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
	// convert the variables from python object to c
	//=========================================================================
	int N[2];
	int permutation1, permutation2;
	PyArrayObject *up1, *up2;
	PyArrayObject *base1, *base2;
	PyArrayObject *psi_up1, *psi_up2;
	PyArrayObject *kapa, *rcm_b, *mrcm_a;
	PyArrayObject *cb1, *cb2;

	N[X_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, X_AXIS));
	N[Y_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Y_AXIS));
	permutation1 = (pml_direction + 1) % 3;
	permutation2 = (pml_direction + 2) % 3;
	up1 = (PyArrayObject *) PyList_GetItem(update_field, permutation1);
	up2 = (PyArrayObject *) PyList_GetItem(update_field, permutation2);
	base1 = (PyArrayObject *) PyList_GetItem(base_field, permutation2);
	base2 = (PyArrayObject *) PyList_GetItem(base_field, permutation1);
	psi_up1 = (PyArrayObject *) PyList_GetItem(psi, pml_direction * 2);
	psi_up2 = (PyArrayObject *) PyList_GetItem(psi, pml_direction * 2 + 1);
	kapa = (PyArrayObject *) PyList_GetItem(pml_coefficient, 0);
	rcm_b = (PyArrayObject *) PyList_GetItem(pml_coefficient, 1);
	mrcm_a = (PyArrayObject *) PyList_GetItem(pml_coefficient,2);
	cb1 = (PyArrayObject *) PyList_GetItem(cb, permutation1);
	cb2 = (PyArrayObject *) PyList_GetItem(cb, permutation2);

	//=========================================================================
	// set variable coefficients
	//=========================================================================
	int i, j;
	int con[2] = {0, 0};  //condition : con_i, con_j
	con[pml_direction] = 1;
	int INorOUT = 1;
	int ijpml = 0;
	int c1[2] = {1, 1};  // ci1, cj1
	int c2[2] = {1, 1};  // ci2, cj2
	int start[2] = {0, 0};  // i_start, j_start
	int end[2] = {N[X_AXIS] + 1, N[Y_AXIS] + 1};  // i_end, j_end
	int pml[2] = {0, 0};  // ipml, jpml

	if (!strcmp(in_out, IN_FIELD))
	{
	   	if (!strcmp(pml_position, FRONT))
	   	{
		   	end[pml_direction] = npml;
	   	}
	   	else
	   	{
		   	ijpml = N[pml_direction] + 1 - 2* npml;
		   	start[pml_direction] = ijpml + npml;
		   	pml[pml_direction] = ijpml;
	   	}
   	}
   	else
   	{
	   	INorOUT *= -1;
	   	for (i = 0; i < 2; i++)
	   	{
		   	c1[i] *= -1;
		   	c2[i] *= -1;
		   	start[i] += 1;
		   	ijpml = 1;
	   	}
	   	if (!strcmp(pml_position, FRONT))
	   	{
		   	end[pml_direction] = npml + 1;
		   	pml[pml_direction] = ijpml;
	   	}
	   	else
	   	{
		   	ijpml = N[pml_direction] + 1 - 2* npml;
		   	start[pml_direction] = ijpml + npml;
		   	pml[pml_direction] = ijpml;
	   	}
   	}

	c2[pml_direction] = 0;
	int cb_EorH = 0;
	for (i = 0; i < 2; i++)
	{
	   	cb_EorH += cb1->dimensions[i];
	   	cb_EorH += cb2->dimensions[i];
   	}
		
   	int c_EH = 0;  // Conditoin of E field or H field
   	int EorH;
   	if (cb_EorH <= 4) 
   	{
	   	c_EH = 0;
	   	EorH = -1;
   	}
   	else
   	{
	   	c_EH = 1;
	   	EorH = 1;
   	}

	float curl_b;  // curl of base fields
	int i_ci1, i_ci2, i_ipml;
	int j_cj1, j_cj2, j_jpml;
	int con_ijpml;
	int c_EH_i, c_EH_j;

	/*
	printf("in_out= %s, INorOUT=%d\n", in_out, INorOUT);
	printf("direction= %d, position= %s\n", pml_direction, pml_position);
	printf("permutation1= %d,%d\n", permutation1, permutation2);
	printf("cb_EorH=%d, c_EH=%d, EorH=%d\n", cb_EorH, c_EH, EorH);
	printf("start=%d,%d,%d, end=%d,%d,%d\n",start[0],start[1],start[2],end[0],end[1],end[2]);
	printf("c1=%d,%d,%d, c2=%d,%d,%d\n", c1[0],c1[1],c1[2],c2[0],c2[1],c2[2]);
	printf("pml=%d,%d,%d, ijpml=%d\n", pml[0],pml[1],pml[2],ijpml);
	printf("con=%d,%d,%d\n\n", con[0],con[1],con[2]);
	*/

	PyArrayObject *up, *base, *psi_up, *cbb;
	if (pml_direction == X_AXIS)
	{
		up = up1;
		base = base1;
		psi_up = psi_up1;
		cbb = cb1;	// avoid the conflict variable name
		EorH *= -1;
	}
	else if (pml_direction == Y_AXIS)
	{
		up = up2;
		base = base2;
		psi_up = psi_up2;
		cbb = cb2;
	}

	//=========================================================================
	// main loop
	//=========================================================================
	for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
	{
	   	for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
	   	{
			i_ci1 = i + c1[X_AXIS];
			i_ci2 = i + c2[X_AXIS];
			j_cj1 = j + c1[Y_AXIS];
			j_cj2 = j + c2[Y_AXIS];
			i_ipml = i - pml[X_AXIS];
			j_jpml = j - pml[Y_AXIS];
			con_ijpml = con[X_AXIS]*i
					+ con[Y_AXIS]*j
					- ijpml;
			c_EH_i = i * c_EH;
			c_EH_j = j * c_EH;
			curl_b = P2F(base, i_ci1, j_cj1)
					- P2F(base, i_ci2, j_cj2);
			//printf("i_ipml=%d, j_jpml=%d", i_ipml, j_jpml);
			P2F(psi_up, i_ipml, j_jpml) = 
					P1F(rcm_b, con_ijpml)
					* P2F(psi_up, i_ipml, j_jpml)
					+ INorOUT
							* P1F(mrcm_a, con_ijpml)
							* curl_b;
			P2F(up, i, j) +=
					+ EorH
					* P2F(cbb, c_EH_i, c_EH_j)
					* (INorOUT
							* (1. / P1F(kapa, con_ijpml) - 1)
							* curl_b
							+ ds*P2F(psi_up, i_ipml, j_jpml));
	   	}
   	}

	//=========================================================================
	// secondary loop for bisymmetry
	//=========================================================================
	if (!strcmp(in_out, OUT_FIELD))
	{
	   	//printf("pml_direction= %d\n",pml_direction);
	   	switch (pml_direction)
	   	{
		   	case X_AXIS:
			   	//printf("Axis= %d\n",X_AXIS);
			   	j = end[Y_AXIS];
			   	j_cj1 = j + c1[Y_AXIS];
			   	j_cj2 = j + c2[Y_AXIS];
			   	c_EH_j = j * c_EH;
			   	for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
			   	{
					i_ci1 = i + c1[X_AXIS];
					i_ci2 = i + c2[X_AXIS];
					i_ipml = i - pml[X_AXIS];
					con_ijpml = con[X_AXIS]*i
							+ con[Y_AXIS]*j
							- ijpml;
					c_EH_i = i * c_EH;
					curl_b = P2F(base, i_ci1, j_cj1)
							- P2F(base, i_ci2, j_cj2);
					P2F(psi_up, i_ipml, 0) = 
							P1F(rcm_b, con_ijpml)
							* P2F(psi_up, i_ipml, 0)
							+ INorOUT
							* P1F(mrcm_a, con_ijpml)
							* curl_b;
					P2F(up, i, j) += 
							+ EorH
							* P2F(cbb, c_EH_i, c_EH_j)
							* (INorOUT
									* (1. / P1F(kapa, con_ijpml) - 1)
									* curl_b
									+ ds*P2F(psi_up, i_ipml, 0));
			   	}
				break;

			 case Y_AXIS:
				//printf("Axis= %d\n",Y_AXIS);
				i = end[X_AXIS];
				i_ci1 = i + c1[X_AXIS];
				i_ci2 = i + c2[X_AXIS];
				c_EH_i = i * c_EH;
				for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
				{
					j_cj1 = j + c1[Y_AXIS];
					j_cj2 = j + c2[Y_AXIS];
					j_jpml = j - pml[Y_AXIS];
					con_ijpml = con[X_AXIS]*i
							+ con[Y_AXIS]*j
							- ijpml;
					c_EH_j = j * c_EH;
					curl_b = P2F(base, i_ci1, j_cj1)
							- P2F(base, i_ci2, j_cj2);
					P2F(psi_up, 0, j_jpml) = 
							P1F(rcm_b, con_ijpml)
							* P2F(psi_up, 0, j_jpml)
							+ INorOUT
							* P1F(mrcm_a, con_ijpml)
							* curl_b;
					P2F(up, i, j) +=
							+ EorH
							* P2F(cbb, c_EH_i, c_EH_j)
							* (INorOUT
									* (1. / P1F(kapa, con_ijpml) - 1)
									* curl_b
									+ ds*P2F(psi_up, 0, j_jpml));
				}
				break;
	   	}
   	}

   	Py_INCREF(Py_None);
   	return Py_None;
}



static PyObject
*update_cpml_1update_2base(PyObject *self, PyObject *args)
{
	//=========================================================================
	// import the python object
	//=========================================================================
	char *in_out;
	PyObject *number_cells, *update_field, *base_field, *cb;
	float ds;
	int npml;
	PyObject *pml_coefficient, *psi;
	int pml_direction;
	char *pml_position;
	if (!PyArg_ParseTuple(args, "sOOOOfiOOis",
			&in_out,
		   	&number_cells,
		   	&update_field,
		   	&base_field,
		   	&cb,
		   	&ds,
		   	&npml,
		   	&pml_coefficient,
		   	&psi,
		   	&pml_direction,
		   	&pml_position))
	{
	   	return NULL;
	}
	//printf("in core: direction = %d, position = %s\n", pml_direction, pml_position);

	//=========================================================================
	// error exception 
	//=========================================================================
	if (strcmp(in_out, IN_FIELD) && strcmp(in_out, OUT_FIELD))
	{
			PyErr_Format(PyExc_ValueError,
						 "In and Out option is one of two strings. in_field is \"in_field\" and out_field is \"out_field\".");
			return NULL;
	}
	if (pml_direction > 2)
	{
			PyErr_Format(PyExc_ValueError,
						 "The information of direction is one of three integers. Zero is the x direction. One is the y direction. Two is the z direction.");
			return NULL;
	}
	if (strcmp(pml_position, FRONT) && strcmp(pml_position, BACK))
	{
			PyErr_Format(PyExc_ValueError,
						 "PML Position option is one of two position. \"front\" or \"back\".");
			return NULL;
	}
	if (!PyList_Check(update_field) && !PyList_Check(base_field) && !PyList_Check(cb) &&
			!PyList_Check(pml_coefficient) && !PyList_Check(psi))
	{
			PyErr_Format(PyExc_ValueError,
						 "Some of arguments are not List.");
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
	// convert the variables from python object to c
	//=========================================================================
	int N[2];
	int permutation1, permutation2;
	PyArrayObject *up1, *up2;
	PyArrayObject *base1, *base2;
	PyArrayObject *psi_up1, *psi_up2;
	PyArrayObject *kapa, *rcm_b, *mrcm_a;
	PyArrayObject *cb1, *cb2;

	N[X_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, X_AXIS));
	N[Y_AXIS] = PyInt_AsLong(PyTuple_GetItem(number_cells, Y_AXIS));
	permutation1 = (pml_direction + 1) % 3;
	permutation2 = (pml_direction + 2) % 3;
	up1 = (PyArrayObject *) PyList_GetItem(update_field, permutation1);
	up2 = (PyArrayObject *) PyList_GetItem(update_field, permutation2);
	base1 = (PyArrayObject *) PyList_GetItem(base_field, permutation2);
	base2 = (PyArrayObject *) PyList_GetItem(base_field, permutation1);
	psi_up1 = (PyArrayObject *) PyList_GetItem(psi, pml_direction * 2);
	psi_up2 = (PyArrayObject *) PyList_GetItem(psi, pml_direction * 2 + 1);
	kapa = (PyArrayObject *) PyList_GetItem(pml_coefficient, 0);
	rcm_b = (PyArrayObject *) PyList_GetItem(pml_coefficient, 1);
	mrcm_a = (PyArrayObject *) PyList_GetItem(pml_coefficient,2);
	cb1 = (PyArrayObject *) PyList_GetItem(cb, permutation1);
	cb2 = (PyArrayObject *) PyList_GetItem(cb, permutation2);

	//=========================================================================
	// set variable coefficients
	//=========================================================================
	int i, j;
	int con[2] = {0, 0};  //condition : con_i, con_j
	con[pml_direction] = 1;
	int INorOUT = 1;
	int ijpml = 0;
	int c1[2] = {1, 1};  // ci1, cj1
	int c2[2] = {1, 1};  // ci2, cj2
	int start[2] = {0, 0};  // i_start, j_start
	int end[2] = {N[X_AXIS] + 1, N[Y_AXIS] + 1};  // i_end, j_end
	int pml[2] = {0, 0};  // ipml, jpml

	if (!strcmp(in_out, IN_FIELD))
	{
	   	if (!strcmp(pml_position, FRONT))
	   	{
		   	end[pml_direction] = npml;
	   	}
	   	else
	   	{
		   	ijpml = N[pml_direction] + 1 - 2* npml;
		   	start[pml_direction] = ijpml + npml;
		   	pml[pml_direction] = ijpml;
	   	}
   	}
   	else
   	{
	   	INorOUT *= -1;
	   	for (i = 0; i < 2; i++)
	   	{
		   	c1[i] *= -1;
		   	c2[i] *= -1;
		   	start[i] += 1;
		   	ijpml = 1;
	   	}
	   	if (!strcmp(pml_position, FRONT))
	   	{
		   	end[pml_direction] = npml + 1;
		   	pml[pml_direction] = ijpml;
	   	}
	   	else
	   	{
		   	ijpml = N[pml_direction] + 1 - 2* npml;
		   	start[pml_direction] = ijpml + npml;
		   	pml[pml_direction] = ijpml;
	   	}
   	}

	c2[pml_direction] = 0;
	int cb_EorH = 0;
	for (i = 0; i < 2; i++)
	{
	   	cb_EorH += cb1->dimensions[i];
	   	cb_EorH += cb2->dimensions[i];
   	}
		
   	int c_EH = 0;  // Conditoin of E field or H field
   	int EorH;
   	if (cb_EorH <= 4)	// required modification for conformal PEC
   	{
	   	c_EH = 0;
	   	EorH = -1;
   	}
   	else
   	{
	   	c_EH = 1;
	   	EorH = 1;
   	}

	float curl_b;  // curl of base fields
	int i_ci1, i_ci2, i_ipml;
	int j_cj1, j_cj2, j_jpml;
	int con_ijpml;
	int c_EH_i, c_EH_j;

	/*
	printf("in_out= %s, INorOUT=%d\n", in_out, INorOUT);
	printf("direction= %d, position= %s\n", pml_direction, pml_position);
	printf("permutation1= %d,%d\n", permutation1, permutation2);
	printf("cb_EorH=%d, c_EH=%d, EorH=%d\n", cb_EorH, c_EH, EorH);
	printf("start=%d,%d,%d, end=%d,%d,%d\n",start[0],start[1],start[2],end[0],end[1],end[2]);
	printf("c1=%d,%d,%d, c2=%d,%d,%d\n", c1[0],c1[1],c1[2],c2[0],c2[1],c2[2]);
	printf("pml=%d,%d,%d, ijpml=%d\n", pml[0],pml[1],pml[2],ijpml);
	printf("con=%d,%d,%d\n\n", con[0],con[1],con[2]);
	*/

	PyArrayObject *up, *base, *psi_up, *cbb;
	if (pml_direction == X_AXIS)
	{
		up = up2;
		base = base2;
		psi_up = psi_up2;
		cbb = cb2;	// avoid the conflict variable name
	}
	else if (pml_direction == Y_AXIS)
	{
		up = up1;
		base = base1;
		psi_up = psi_up1;
		cbb = cb1;
		EorH *= -1;
	}

	//=========================================================================
	// main loop
	//=========================================================================
	for (i = start[X_AXIS]; i < end[X_AXIS]; i++)
	{
	   	for (j = start[Y_AXIS]; j < end[Y_AXIS]; j++)
	   	{
			i_ci1 = i + c1[X_AXIS];
			i_ci2 = i + c2[X_AXIS];
			j_cj1 = j + c1[Y_AXIS];
			j_cj2 = j + c2[Y_AXIS];
			i_ipml = i - pml[X_AXIS];
			j_jpml = j - pml[Y_AXIS];
			con_ijpml = con[X_AXIS]*i
					+ con[Y_AXIS]*j
					- ijpml;
			c_EH_i = i * c_EH;
			c_EH_j = j * c_EH;
			curl_b = P2F(base, i_ci1, j_cj1)
					- P2F(base, i_ci2, j_cj2);
			//printf("i_ipml=%d, j_jpml=%d\n", i_ipml, j_jpml);
			P2F(psi_up, i_ipml, j_jpml) = 
					P1F(rcm_b, con_ijpml)
					* P2F(psi_up, i_ipml, j_jpml)
					+ INorOUT
							* P1F(mrcm_a, con_ijpml)
							* curl_b;
			P2F(up, i, j) += 
					+ EorH
					* P2F(cbb, c_EH_i, c_EH_j)
					* (INorOUT
							* (1. / P1F(kapa, con_ijpml) - 1)
							* curl_b
							+ ds*P2F(psi_up, i_ipml, j_jpml));
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
static char update_cpml_3d_doc[] = "update_cpml_3d(in_out_field, number_cells, update_field, base_field, cb, ds, npml, pml_coefficient, psi, pml_direction, pml_position)";
static char update_cpml_2update_1base_doc[] = "update_cpml_2update_1base(in_out_field, number_cells, update_field, base_field, cb, ds, npml, pml_coefficient, psi, pml_direction, pml_position)";
static char update_cpml_1update_2base_doc[] = "update_cpml_1update_2base(in_out_field, number_cells, update_field, base_field, cb, ds, npml, pml_coefficient, psi, pml_direction, pml_position)";
static char module_doc[] = \
	"module cpml_core:\n\
	update_cpml_3d(in_out_field, number_cells, update_field, base_field, cb, ds, npml, pml_coefficient, psi, pml_direction, pml_position)\n\
	update_cpml_2update_1base(in_out_field, number_cells, update_field, base_field, cb, ds, npml, pml_coefficient, psi, pml_direction, pml_position)\n\
	update_cpml_1update_2base(in_out_field, number_cells, update_field, base_field, cb, ds, npml, pml_coefficient, psi, pml_direction, pml_position)";

static PyMethodDef cpml_core_methods[] = {
	{"update_cpml_3d", update_cpml_3d, METH_VARARGS, update_cpml_3d_doc},
	{"update_cpml_2update_1base", update_cpml_2update_1base, METH_VARARGS, update_cpml_2update_1base_doc},
	{"update_cpml_1update_2base", update_cpml_1update_2base, METH_VARARGS, update_cpml_1update_2base_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcpml_core() {
	Py_InitModule3("cpml_core", cpml_core_methods, module_doc);
	import_array();
}
