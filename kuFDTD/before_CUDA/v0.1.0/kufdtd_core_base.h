/*
 <File Description>

 File Name : matter_core_base.h

 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Kim Ki-hwan (wbkifun@korea.ac.kr) 

 Written date : 2008. 2. 1. Fri

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the global variables for c core

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
#define P2F(a, i, j)\
        *((float *)(a->data + (i) * a->strides[0] + (j) * a->strides[1] ))
#define P2D(a, i, j)\
        *((double *)(a->data + (i) * a->strides[0] + (j) * a->strides[1] ))
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
