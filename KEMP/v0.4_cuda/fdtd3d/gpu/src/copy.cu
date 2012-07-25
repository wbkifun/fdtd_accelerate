__global__ void copy(int nx, int ny, int nz, int shift_idx, DTYPE *target, ARGS) { 
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx, sub_idx;

    while( gid < NMAX ) {
        idx = XID*ny*nz + YID*nz + ZID;
        sub_idx = gid + NMAX*shift_idx;

        TARGET OVERWRITE SOURCE;
        gid += blockDim.x * gridDim.x;
    }
} 
