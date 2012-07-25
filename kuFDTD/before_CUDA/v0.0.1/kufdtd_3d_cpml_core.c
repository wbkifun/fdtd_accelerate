/*
 <File Description>

 File Name : kufdtd_3d_cpml.c

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 01. 15. Tue

 Copyright : This has used lots of python modules which is opend to public. So,
 it is also in pulic.

============================== < File Description > ===========================

이 파일은 KUFDTD(Korea University Finite Difference Time Domain method)의
 3D 계산을 위해 필요한 핵심 코어 함수들을 모아놓은 파일이다. 이 파일 안에는
 3D cpml calculation들이
 들어있다.

===============================================================================
*/

#include <kufdtd_core_base.h>

static PyObject
*update_cpml(PyObject *self, PyObject *args)
{

        char *in_out;
        char *pml_position;
        int N[3];
        int info_direction;
        int npml;
        float ds;
        PyArrayObject *up1, *up2;
        PyArrayObject *base1, *base2;
        PyArrayObject *psi_up1, *psi_up2;
        PyArrayObject *kapa, *rcm_b, *mrcm_a;
        PyArrayObject *cb1, *cb2;
        PyObject *number_cells, *update_field, *base_field, *psi;
        PyObject *pml_parameter, *cb;

        if (!PyArg_ParseTuple(args, "ssiifOOOOOO",
                              &pml_position,
                              &in_out,
                              &info_direction,
                              &npml,
                              &ds,
                              &number_cells,
                              &update_field,
                              &base_field,
                              &psi,
                              &pml_parameter,
                              &cb))
        {
                return NULL;
        }
        if (!strcmp(pml_position, FRONT) || !strcmp(pml_position, BACK))
        {
                PyErr_Formar(PyExc_ValueError,
                             "PML Position option is one of two position.\
                             front is \"front\" and back is \"back\".");
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
        if (info_direction > 2)
        {
                PyErr_Format(PyExc_ValueError,
                             "The information of direction is one of\
                             three integers.\
                             Zero is the x direction. One is the y direction.\
                             Two is the z direction.");
                return NULL;
        }
        if (!PyList_Check(number_cells) && !PyList_Check(update_field) &&
                !PyList_Check(base_field) &&  !PyList_Check(psi) &&
                !PyList_Check(pml_parameter) && !PyList_Check(cb))
        {
                PyErr_Format(PyExc_ValueError,
                             "Some of arguments are not List, except,\
                             info_fieldface, in_out, ds.");
                return NULL;
        }
        N[X_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, X_AXIS));
        N[Y_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, Y_AXIS));
        N[Z_AXIS] = PyInt_AsInt(PyList_GetItem(number_cells, Z_AXIS));
        int permutation1, permutation2;
        permutation1 = (info_direction + 1) % 3;
        permutation2 = (info_direction + 2) % 3;
        base1 = (PyArrayObject *) PyList_GetItem(base_field, permutation2);
        base2 = (PyArrayObject *) PyList_GetItem(base_field, permutation1);
        up1 = (PyArrayObject *) PyList_GetItem(update_field, permutation1);
        up2 = (PyArrayObject *) PyList_GetItem(update_field, permutation2);
        psi_up1 = (PyArrayObject *) PyList_GetItem(psi, info_direction * 2);
        psi_up2 = (PyArrayObject *) PyList_GetItem(psi, info_direction * 2 + 1);
        kapa = (PyArrayObject *) PyList_GetItem(pml_parameter, 0);
        rcm_b = (PyArrayObject *) PyList_GetItem(pml_parameter, 1);
        mrcm_a = (PyArrayObject *) PyList_GetItem(pml_parameter,2);
        cb1 = (PyArrayObject *) PyList_GetItem(cb, permutation1);
        cb2 = (PyArrayObject *) PyList_GetItem(cb, permutation2);

        int i, j, k;
        int con[3] = {0, 0, 0};  //condition : con_i, con_j, con_k
        con[info_direction] = 1;
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
                        end[info_direction] = npml;
                }
                else
                {
                        ijkpml = N[info_direction] + 1 - 2* npml;
                        start[info_direction] = ijkpml + npml;
                        pml[info_direction] = ijkpml;
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
                        ijkpml += 1;
                }
                if (!strcmp(pml_position, FRONT))
                {
                        end[info_direction] = npml + 1;
                        pml[info_direction] = ijkpml;
                }
                else
                {
                        ijkpml = N[info_direction] + 1 - 2* npml;
                        start[info_direction] = ijkpml + npml;
                        pml[info_direction] = ijkpml;
                }
        }
        c2[info_direction] = 0;
        int cb_EorH = 0;
        for (i = 0; i < 3; i++)
        {
                cb_EorH += cb1->dimensions[i];
                cb_EorH += cb2->dimensions[i];
        }
        int c_EH = 0;  // Conditoin of E field or H field
        int EorH;
        if (cb_EorH <= 6)
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
                               con_ijkpml =
                                         con[X_AXIS] * i
                                       + con[Y_AXIS] * j
                                       + con[Z_AXIS] * k
                                       - ijkpml;
                               c_EH_i = i * c_EH;
                               c_EH_j = j * c_EH;
                               c_EH_k = k * c_EH;
                               curl_b1 =
                                    P3F(base1, i_ci1, j_cj1, k_ck1)
                                  - P3F(base1, i_ci2, j_cj2, k_ck2);
                               curl_b2 =
                                    P3F(base2, i_ci1, j_cj1, k_ck1)
                                  - P3F(base2, i_ci2, j_cj2, k_ck2);
                               P3F(psi_up1, i_ipml, j_jpml, k_kpml) =
                                    P1F(rcm_b, con_ijkpml)
                                  * P3F(psi_up1, i_ipml, j_jpml, k_kpml)
                                  + INorOUT
                                  * P1F(mrcm_a, con_ijkpml)
                                  * curl_b1 / ds;
                               P3F(psi_up2, i_ipml, j_jpml, k_kpml) =
                                    P1F(rcm_b, con_ijkpml)
                                  * P3F(psi_up2, i_ipml, j_jpml, k_kpml)
                                  + INorOUT
                                  * P1F(mrcm_a, con_ijkpml)
                                  * curl_b2 / ds;
                               P3F(up1, i, j, k) +=
                                  - EorH
                                  * P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
                                  * (INorOUT
                                  * (1. / P1F(kapa, con_ijkpml) - 1)
                                  * curl_b1
                                  + ds
                                  * P3F(psi_up1, i_ipml, j_jpml, k_kpml));
                               P3F(up2, i, j, k) +=
                                  + EorH
                                  * P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
                                  * (INorOUT
                                  * (1. / P1F(kapa, con_ijkpml) - 1)
                                  * curl_b2
                                  + ds
                                  * P3F(psi_up2, i_ipml, j_jpml, k_kpml));
                        }
                }
        }
        if (!strcmp(in_out, OUT_FIELD))
        {
                switch (info_direction)
                {
                case X_AXIS:
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
                                       con_ijkpml =
                                                 con[X_AXIS] * i
                                               + con[Y_AXIS] * j
                                               + con[Z_AXIS] * k
                                               - ijkpml;
                                       c_EH_i = i * c_EH;
                                       c_EH_k = k * c_EH;
                                       curl_b1 =
                                            P3F(base1, i_ci1, j_cj1, k_ck1)
                                          - P3F(base1, i_ci2, j_cj2, k_ck2);
                                       P3F(psi_up1, i_ipml, 0, k_kpml) =
                                            P1F(rcm_b, con_ijkpml)
                                          * P3F(psi_up1, i_ipml, 0, k_kpml)
                                          + INorOUT
                                          * P1F(mrcm_a, con_ijkpml)
                                          * curl_b1 / ds;
                                       P3F(up1, i, j, k) +=
                                          - EorH
                                          * P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
                                          * (INorOUT
                                          * (1. / P1F(kapa, con_ijkpml) - 1)
                                          * curl_b1
                                          + ds
                                          * P3F(psi_up1, i_ipml, 0, k_kpml));
                                }
                        }
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
                                       con_ijkpml =
                                                 con[X_AXIS] * i
                                               + con[Y_AXIS] * j
                                               + con[Z_AXIS] * k
                                               - ijkpml;
                                       c_EH_i = i * c_EH;
                                       c_EH_j = j * c_EH;
                                       curl_b2 =
                                            P3F(base2, i_ci1, j_cj1, k_ck1)
                                          - P3F(base2, i_ci2, j_cj2, k_ck2);
                                       P3F(psi_up2, i_ipml, j_jpml, 0) =
                                            P1F(rcm_b, con_ijkpml)
                                          * P3F(psi_up2, i_ipml, j_jpml, 0)
                                          + INorOUT
                                          * P1F(mrcm_a, con_ijkpml)
                                          * curl_b2 / ds;
                                       P3F(up2, i, j, k) +=
                                          + EorH
                                          * P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
                                          * (INorOUT
                                          * (1. / P1F(kapa, con_ijkpml) - 1)
                                          * curl_b2
                                          + ds
                                          * P3F(psi_up2, i_ipml, j_jpml, 0));
                                }
                        }
                 case Y_AXIS:
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
                                       con_ijkpml =
                                                 con[X_AXIS] * i
                                               + con[Y_AXIS] * j
                                               + con[Z_AXIS] * k
                                               - ijkpml;
                                       c_EH_i = i * c_EH;
                                       c_EH_j = j * c_EH;
                                       curl_b1 =
                                            P3F(base1, i_ci1, j_cj1, k_ck1)
                                          - P3F(base1, i_ci2, j_cj2, k_ck2);
                                       P3F(psi_up1, i_ipml, j_jpml, 0) =
                                            P1F(rcm_b, con_ijkpml)
                                          * P3F(psi_up1, i_ipml, j_jpml, 0)
                                          + INorOUT
                                          * P1F(mrcm_a, con_ijkpml)
                                          * curl_b1 / ds;
                                       P3F(up1, i, j, k) +=
                                          - EorH
                                          * P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
                                          * (INorOUT
                                          * (1. / P1F(kapa, con_ijkpml) - 1)
                                          * curl_b1
                                          + ds
                                          * P3F(psi_up1, i_ipml, j_jpml, 0));
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
                                       con_ijkpml =
                                                 con[X_AXIS] * i
                                               + con[Y_AXIS] * j
                                               + con[Z_AXIS] * k
                                               - ijkpml;
                                       c_EH_j = j * c_EH;
                                       c_EH_k = k * c_EH;
                                       curl_b2 =
                                            P3F(base2, i_ci1, j_cj1, k_ck1)
                                          - P3F(base2, i_ci2, j_cj2, k_ck2);
                                       P3F(psi_up2, 0, j_jpml, k_kpml) =
                                            P1F(rcm_b, con_ijkpml)
                                          * P3F(psi_up2, 0, j_jpml, k_kpml)
                                          + INorOUT
                                          * P1F(mrcm_a, con_ijkpml)
                                          * curl_b2 / ds;
                                       P3F(up2, i, j, k) +=
                                          + EorH
                                          * P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
                                          * (INorOUT
                                          * (1. / P1F(kapa, con_ijkpml) - 1)
                                          * curl_b2
                                          + ds
                                          * P3F(psi_up2, 0, j_jpml, k_kpml));
                                }
                        }
                 case Z_AXIS:
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
                                       con_ijkpml =
                                                 con[X_AXIS] * i
                                               + con[Y_AXIS] * j
                                               + con[Z_AXIS] * k
                                               - ijkpml;
                                       c_EH_j = j * c_EH;
                                       c_EH_k = k * c_EH;
                                       curl_b1 =
                                            P3F(base1, i_ci1, j_cj1, k_ck1)
                                          - P3F(base1, i_ci2, j_cj2, k_ck2);
                                       P3F(psi_up1, 0, j_jpml, k_kpml) =
                                            P1F(rcm_b, con_ijkpml)
                                          * P3F(psi_up1, 0, j_jpml, k_kpml)
                                          + INorOUT
                                          * P1F(mrcm_a, con_ijkpml)
                                          * curl_b1 / ds;
                                       P3F(up1, i, j, k) +=
                                          - EorH
                                          * P3F(cb1, c_EH_i, c_EH_j, c_EH_k)
                                          * (INorOUT
                                          * (1. / P1F(kapa, con_ijkpml) - 1)
                                          * curl_b1
                                          + ds
                                          * P3F(psi_up1, 0, j_jpml, k_kpml));
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
                                       con_ijkpml =
                                                 con[X_AXIS] * i
                                               + con[Y_AXIS] * j
                                               + con[Z_AXIS] * k
                                               - ijkpml;
                                       c_EH_i = i * c_EH;
                                       c_EH_k = k * c_EH;
                                       curl_b2 =
                                            P3F(base2, i_ci1, j_cj1, k_ck1)
                                          - P3F(base2, i_ci2, j_cj2, k_ck2);
                                       P3F(psi_up2, i_ipml, 0, k_kpml) =
                                            P1F(rcm_b, con_ijkpml)
                                          * P3F(psi_up2, i_ipml, 0, k_kpml)
                                          + INorOUT
                                          * P1F(mrcm_a, con_ijkpml)
                                          * curl_b2 / ds;
                                       P3F(up2, i, j, k) +=
                                          + EorH
                                          * P3F(cb2, c_EH_i, c_EH_j, c_EH_k)
                                          * (INorOUT
                                          * (1. / P1F(kapa, con_ijkpml) - 1)
                                          * curl_b2
                                          + ds
                                          * P3F(psi_up2, i_ipml, 0, k_kpml));
                                }
                        }
                }
        }
        Py_INCREF(Py_None);
        return Py_None;
}
