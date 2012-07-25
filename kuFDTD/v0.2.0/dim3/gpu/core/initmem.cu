/*
 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Ki-Hwan Kim (wbkifun@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
*/

__global__ void initmem( int Ntot, int idx0, float *a ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x + idx0;

	if ( idx < Ntot ) a[idx] = 0;
}
