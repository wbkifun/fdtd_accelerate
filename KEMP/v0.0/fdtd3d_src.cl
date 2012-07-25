__kernel void update(const float tstep, __global float *f) {
	const int idx = get_global_id(0);
	//int ijk = (NX/2)*NYZ + (NY/2)*NZ + idx;
	int ijk = (2*NX/3)*NYZ + (NY/2)*NZ + idx;

	if( idx < NZ ) f[ijk] += sin(0.1*tstep);
}
