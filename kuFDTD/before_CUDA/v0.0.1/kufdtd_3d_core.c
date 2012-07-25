/*
 <File Description>

 File Name : kufdtd_3d_core.c

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 01. 15. Tue

 Copyright : This has used lots of python modules which is opend to public. So,
 it is also in pulic.

============================== < File Description > ===========================

이 파일은 KUFDTD(Korea University Finite Difference Time Domain method)의
 3D 계산을 위해 필요한 핵심 코어 함수들을 모아놓은 파일이다. 이 파일 안에는
 3D dielectric calculation, 3D metal calculation들이
 들어있다.

===============================================================================
*/

#include <kufdtd_core_base.h>

static PyObject
*update_non_dispersive(PyObject *self, PyObject *args)
{
        char *info_fieldface;
        char *in_out;
        int N[3];
        PyArrayObject *upx, *upy, *upz;
        PyArrayObject *basex, *basey, *basez;
        PyArrayObject *cbx, *cby, *cbz;
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
        if (!strcmp(in_out, IN_FIELD) || !strcmp(in_out, OUT_FIELD))
        {
                PyErr_Format(PyExc_ValueError,
                             "In and Out option is one of two strings.\
                             in_field is \"in_field\" and out_field is \
                             \"out_field\".");
                return NULL;
        }
        if (!strcmp(info_fieldface, EFACED) || !strcmp(info_fieldface, HFACED))
        {
                PyErr_Format(PyExc_ValueError,
                             "The information of field face is one of\
                             two strings.\
                             efaced is \"efaced\" and hfaced is \"hfaced\".");
                return NULL;
        }
        if (!PyList_Check(number_cells) && !PyList_Check(update_field) &&
                !PyList_Check(base_field) && !PyList_Check(cb))
        {
                PyErr_Format(PyExc_ValueError,
                             "Some of arguments are not List, except,\
                             info_fieldface and in_out.");
                return NULL;
        }
        N[X_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, X_AXIS));
        N[Y_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, Y_AXIS));
        N[Z_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, Z_AXIS));
        upx = (PyArrayObject *) PyList_GetItem(update_field, X_AXIS);
        upy = (PyArrayObject *) PyList_GetItem(update_field, Y_AXIS);
        upz = (PyArrayObject *) PyList_GetItem(update_field, Z_AXIS);
        basex = (PyArrayObject *) PyList_GetItem(base_field, X_AXIS);
        basey = (PyArrayObject *) PyList_GetItem(base_field, Y_AXIS);
        basez = (PyArrayObject *) PyList_GetItem(base_field, Z_AXIS);
        cbx = (PyArrayObject *) PyList_GetItem(cb, X_AXIS);
        cby = (PyArrayObject *) PyList_GetItem(cb, Y_AXIS);
        cbz = (PyArrayObject *) PyList_GetItem(cb, Z_AXIS);

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
                               curl_bzy =
                                    P3F(basez, i_c1, j_c1, k_c1)
                                  - P3F(basez, i_c1, j   , k_c1);
                               curl_byz =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i_c1, j_c1, k   );
                               curl_bxz =
                                    P3F(basex, i_c1, j_c1, k_c1)
                                  - P3F(basex, i_c1, j_c1, k   );
                               curl_bzx =
                                    P3F(basez, i_c1, j_c1, k_c1)
                                  - P3F(basez, i   , j_c1, k_c1);
                               curl_byx =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i   , j_c1, k_c1);
                               curl_bxy =
                                    P3F(basex, i_c1, j_c1, k_c1)
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
                               curl_bzy =
                                    P3F(basez, i_c1, j_c1, k_c1)
                                  - P3F(basez, i_c1, j   , k_c1);
                               curl_byz =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i_c1, j_c1, k   );
                               P3F(upx, i, j, k) +=
                                       P3F(cbx, c_EH_i, c_EH_j, c_EH_k)
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
                               curl_bxz =
                                    P3F(basex, i_c1, j_c1, k_c1)
                                  - P3F(basex, i_c1, j_c1, k   );
                               curl_bzx =
                                    P3F(basez, i_c1, j_c1, k_c1)
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
                               curl_byx =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i   , j_c1, k_c1);
                               curl_bxy =
                                    P3F(basex, i_c1, j_c1, k_c1)
                                  - P3F(basex, i_c1, j   , k_c1);
                               P3F(upz, i, j, k) +=
                                       P3F(cbz, c_EH_i, c_EH_j, c_EH_k)
                                       * info_EorHfaced
                                       * (curl_byx - curl_bxy);
                        }
                }
        }
        Py_INCREF(Py_None);
        return Py_None;
}

static PyObject
*update_drude_metal(PyObject *self, PyObject *args)
{
        char *info_fieldface;
        char *in_out;
        int N[3];
        float dt;
        PyArrayObject *upx, *upy, *upz;
        PyArrayObject *basex, *basey, *basez;
        PyArrayObject *fupx, *fupy, *fupz;
        PyArrayObject *gamma_upx, *gamma_upy, *gamma_upz;
        PyArrayObject *cax, *cay, *caz;
        PyArrayObject *cbx, *cby, *cbz;
        PyArrayObject *cfx, *cfy, *cfz;
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
        if (!strcmp(in_out, IN_FIELD) || !strcmp(in_out, OUT_FIELD))
        {
                PyErr_Format(PyExc_ValueError,
                             "In and Out option is one of two strings.\
                             in_field is \"in_field\" and out_field is \
                             \"out_field\".");
                return NULL;
        }
        if (!strcmp(info_fieldface, EFACED) || !strcmp(info_fieldface, HFACED))
        {
                PyErr_Format(PyExc_ValueError,
                             "The information of field face is one of\
                             two strings.\
                             efaced is \"efaced\" and hfaced is \"hfaced\".");
                return NULL;
        }
        if (!PyList_Check(number_cells) && !PyList_Check(update_field)
                && !PyList_Check(base_field) && !PyList_Check(update_f_field)
                && !PyList_Check(gamma_up) && !PyList_Check(ca)
                &&!PyList_Check(cb) && !PyList_Check(cf))
        {
                PyErr_Format(PyExc_ValueError,
                             "Some of arguments are not List, except,\
                             info_fieldface, in_out, dt.");
                return NULL;
        }
        N[X_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, X_AXIS));
        N[Y_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, Y_AXIS));
        N[Z_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, Z_AXIS));
        upx = (PyArrayObject *) PyList_GetItem(update_field, X_AXIS);
        upy = (PyArrayObject *) PyList_GetItem(update_field, Y_AXIS);
        upz = (PyArrayObject *) PyList_GetItem(update_field, Z_AXIS);
        basex = (PyArrayObject *) PyList_GetItem(base_field, X_AXIS);
        basey = (PyArrayObject *) PyList_GetItem(base_field, Y_AXIS);
        basez = (PyArrayObject *) PyList_GetItem(base_field, Z_AXIS);
        fupx = (PyArrayObject *) PyList_GetItem(update_f_field, X_AXIS);
        fupy = (PyArrayObject *) PyList_GetItem(update_f_field, Y_AXIS);
        fupz = (PyArrayObject *) PyList_GetItem(update_f_field, Z_AXIS);
        cax = (PyArrayObject *) PyList_GetItem(ca, X_AXIS);
        cay = (PyArrayObject *) PyList_GetItem(ca, Y_AXIS);
        caz = (PyArrayObject *) PyList_GetItem(ca, Z_AXIS);
        cbx = (PyArrayObject *) PyList_GetItem(cb, X_AXIS);
        cby = (PyArrayObject *) PyList_GetItem(cb, Y_AXIS);
        cbz = (PyArrayObject *) PyList_GetItem(cb, Z_AXIS);
        cfx = (PyArrayObject *) PyList_GetItem(cf, X_AXIS);
        cfy = (PyArrayObject *) PyList_GetItem(cf, Y_AXIS);
        cfz = (PyArrayObject *) PyList_GetItem(cf, Z_AXIS);

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
        for (i = i_start; i < i_end; i++)
        {
                for (j = j_start; j < j_end; j++)
                {
                        for (k = k_start; k < k_end; k++)
                        {
                               i_c1 = i + c1;
                               j_c1 = j + c1;
                               k_c1 = k + c1;
                               curl_bzy =
                                    P3F(basez, i_c1, j_c1, k_c1)
                                  - P3F(basez, i_c1, j   , k_c1);
                               curl_byz =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i_c1, j_c1, k   );
                               curl_bxz =
                                    P3F(basex, i_c1, j_c1, k_c1)
                                  - P3F(basex, i_c1, j_c1, k   );
                               curl_bzx =
                                    P3F(basez, i_c1, j_c1, k_c1)
                                  - P3F(basez, i   , j_c1, k_c1);
                               curl_byx =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i   , j_c1, k_c1);
                               curl_bxy =
                                    P3F(basex, i_c1, j_c1, k_c1)
                                  - P3F(basex, i_c1, j   , k_c1);
                               P3F(fupx, i, j, k) =
                                       dt * P3F(upx, i, j, k)
                                       + exp(-dt * P3F(gamma_upx, i, j, k))
                                       * P3F(fupx, i, j, k);
                               P3F(fupy, i, j, k) =
                                       dt * P3F(upy, i, j, k)
                                       + exp(-dt * P3F(gamma_upy, i, j, k))
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
                               curl_bzy =
                                    P3F(basez, i_c1, j_c1, k_c1)
                                  - P3F(basez, i_c1, j   , k_c1);
                               curl_byz =
                                    P3F(basey, i_c1, j_c1, k_c1)
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
                               curl_bxz =
                                    P3F(basex, i_c1, j_c1, k_c1)
                                  - P3F(basex, i_c1, j_c1, k   );
                               curl_bzx =
                                    P3F(basez, i_c1, j_c1, k_c1)
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
                               curl_byx =
                                    P3F(basey, i_c1, j_c1, k_c1)
                                  - P3F(basey, i   , j_c1, k_c1);
                               curl_bxy =
                                    P3F(basex, i_c1, j_c1, k_c1)
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
