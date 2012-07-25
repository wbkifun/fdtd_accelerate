PRAGMA_fp64

__kernel void copy(int nx, int ny, int nz, __global DTYPE *f) {
	int gid = get_global_id(0);
    int idx0, idx1;

	while( gid < NMAX ) {
        idx0 = IDX0;
        idx1 = IDX1;

	   	f[idx1] = f[idx0];
		gid += get_global_size(0);
	}
} 
