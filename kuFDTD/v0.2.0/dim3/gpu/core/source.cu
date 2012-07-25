/*
 Author : Kim, KyoungHo (rain_woo@korea.ac.kr)
          Ki-Hwan Kim (wbkifun@korea.ac.kr)

 Written date : 2009. 6. 11
 last update  : 

 Copyright : GNU GPL
*/

__global__ void update_src( int Nx, int Ny, int Nz, int tstep, float *F ) {
	int idx, ijk;
	idx = threadIdx.x;
	//ijk = (idx+1)*Ny*Nz + (Ny/2)*Nz + (Nz/2);
	//ijk = (idx+1)*Ny*Nz + (Ny/2 - 30)*Nz + (Nz/2 - 50);
	//ijk = (Nx/2 - 30)*Ny*Nz + (idx)*Nz + (Nz/2 - 50);
	ijk = (Nx/2-30)*Ny*Nz + (Ny/2-50)*Nz + idx;

	F[ijk] += sin(0.1*tstep);
}

