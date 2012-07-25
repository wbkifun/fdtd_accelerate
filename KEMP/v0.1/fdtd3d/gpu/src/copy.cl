__kernel void copy(int nx, int ny, int nz, __global float *f) {
	int gid = get_global_id(0);

	while( gid < NMAX ) {
	   	f[IDX1] = f[IDX2];
		gid += get_global_size(0);
	}
} 
