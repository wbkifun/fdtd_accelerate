/*
 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Ki-Hwan Kim (wbkifun@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
*/

#include <Python.h>
#include <numpy/arrayobject.h>
//#include <omp.h>

#define LOAD __builtin_ia32_loadups
#define STORE __builtin_ia32_storeups
#define ADD __builtin_ia32_addps
#define SUB __builtin_ia32_subps
#define MUL __builtin_ia32_mulps

typedef float v4sf __attribute__ ((vector_size(16)));

union f4vector {
	v4sf v;
	float f[4];
};
