__kernel void update_h(__global const float *ex, __global const float *ey, __global const float *ez, __global float *hx, __global float *hy, __global float *hz) {
	const int tx = get_local_id(0);
	int idx = get_global_id(0);

	__local float s[3*DX+2];
	__local float *sx, *sy, *sz;
	sz = s;
	sy = &sz[DX+1];
	sx = &sy[DX+1];

	while( idx < NXYZ ) {
		sz[tx] = ez[idx];
		sy[tx] = ey[idx];
		sx[tx] = ex[idx];
		if( tx == 0 ) {
			sy[tx-1] = ey[idx-1];
			sx[tx-1] = ex[idx-1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		int i = idx/(NYZ);
		int j = (idx - i*NYZ)/NZ;
		int k = idx%NZ;

		if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[tx] - ez[idx-NZ] - sy[tx] + sy[tx-1] );
		if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[tx] - sx[tx-1] - sz[tx] + ez[idx-NYZ] );
		if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[tx] - ey[idx-NYZ] - sx[tx] + ex[idx-NZ] );

		idx += get_global_size(0);
	}
}

__kernel void update_e(__global float *ex, __global float *ey, __global float *ez, __global const float *hx, __global const float *hy, __global const float *hz, __global const float *cex, __global const float *cey, __global const float *cez) {
	const int tx = get_local_id(0);
	int idx = get_global_id(0);
	
	__local float sx[DX+1], sy[DX+1], sz[DX];

	while( idx < NXYZ ) {
		sz[tx] = hz[idx];
		sy[tx] = hy[idx];
		sx[tx] = hx[idx];
		if( tx == DX-1 ) {
			sy[tx+1] = hy[idx+1];
			sx[tx+1] = hx[idx+1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		int i = idx/(NYZ);
		int j = (idx - i*NYZ)/NZ;
		int k = idx%NZ;

		if( j<NY-1 && k<NZ-1 ) ex[idx] += cex[idx]*( hz[idx+NZ] - sz[tx] - sy[tx+1] + sy[tx] );
		if( i<NX-1 && k<NZ-1 ) ey[idx] += cey[idx]*( sx[tx+1] - sx[tx] - hz[idx+NYZ] + sz[tx] );
		if( i<NX-1 && j<NY-1 ) ez[idx] += cez[idx]*( hy[idx+NYZ] - sy[tx] - hx[idx+NZ] + sx[tx] );

		idx += get_global_size(0);
	}
}
