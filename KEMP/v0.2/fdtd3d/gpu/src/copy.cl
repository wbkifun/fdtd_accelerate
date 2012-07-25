PRAGMA_fp64

__kernel void subdomain(int nx, int ny, int nz, int shift_idx, __global DTYPE *target, ARGS) { 
    int gid = get_global_id(0);
    int idx, sub_idx;

    while( gid < NMAX ) {
        idx = XID*ny*nz + YID*nz + ZID;
        sub_idx = gid + NMAX*shift_idx;

        TARGET OVERWRITE SOURCE;
        gid += get_global_size(0);
    }
} 
