/*
 <File Description>

 File Name : kufdtd_3d_efaced_dielectric_core.c

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Written date : 2008. 01. 08. Tue

 Modifier : Kim, KyoungHo (rain_woo@korea.ac.kr)
 Modified date : 2008. 01. 08. Tue

 Copyright : This has used lots of python modules which is opend to public. So,
 it is also in pulic.

============================== < File Description > ===========================

이 파일은 KUFDTD(Korea University Finite Difference Time Domain method)의
 3D 계산을 위해 필요한 핵심 코어 함수들을 모아놓은 파일이다. 이 파일 안에는
 3D dielectric calculation, 3D metal calculation, 3D cpml calculation class들이
 들어있다.

===============================================================================
*/

#ifndef __KUFDTD_CORE_BASE_H
#define __KUFDTD_CORE_BASE_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#define P1F(a, i)\
        *((float *)(a->data + (i) * a->strides[0]))
#define P1D(a, i)\
        *((double *)(a->data + (i) * a->strides[0]))
#define P3F(a, i, j, k)\
        *((float *)(a->data + (i) * a->strides[0]\
                        + (j) * a->strides[1] + (k) * a->strides[2]))
#define P3D(a, i, j, k)\
        *((double *)(a->data + (i) * a->strides[0]\
                        + (j) * a->strides[1] + (k) * a->strides[2]))
#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2
#define IN_FIELD "in_field"
#define OUT_FIELD "out_field"
#define EFACED "efaced"
#define HFACED "hfaced"
#define FRONT "front"
#define BACK "back"
#endif
