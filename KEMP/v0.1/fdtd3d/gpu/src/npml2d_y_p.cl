__kernel void update_e(int nx, int ny, int nz, 
		__global float *e0, __global float *e1, 
		__global float *h0, __global float *h1, 
		__global float *pe0, __global float *pe1,
		__global float *ph0, __global float *ph1, 
		float cb0, 
		float ca1, float cb1,
	   	float ca2, float cb2) {

	int gid = get_global_id(0);
	int idx1, idx2, gid2;
	float e0_new, e1_new;

	while( gid < nx*nz ) {
		idx1 = (gid/nz)*ny*nz + (ny-1)*nz + gid%nz;
		idx2 = (gid/nz)*ny*nz + (ny-2)*nz + gid%nz;
		gid2 = gid + nx*nz;

		e0_new = e0[idx2] - 0.5 * (ph1[gid] - ph1[gid2]);
		e1_new = e1[idx2] + 0.5 * (ph0[gid] - ph0[gid2]);
		e0[idx2] = e0_new;
		e1[idx2] = e1_new;

		pe0[gid] -= cb0 * e0_new;
		pe1[gid] -= cb0 * e1_new;

		ph0[gid] = ca1 * ph0[gid] - cb1 * h0[idx1];
		ph1[gid] = ca1 * ph1[gid] - cb1 * h1[idx1];
		ph0[gid2] = ca2 * ph0[gid2] - cb2 * h0[idx2];
		ph1[gid2] = ca2 * ph1[gid2] - cb2 * h1[idx2];

		gid += get_global_size(0);
	}
} 



__kernel void update_h(int nx, int ny, int nz,
	   	__global float *e0, __global float *e1,
	   	__global float *h0, __global float *h1,
	   	__global float *pe0, __global float *pe1,
		__global float *ph0, __global float *ph1,
	   	float ca0, float cb0,
	   	float cb1, 
		float cb2) {

	int gid = get_global_id(0);
	int idx1, idx2, gid2;
	float h0_new, h1_new;

	while( gid < nx*nz ) {
		idx1 = (gid/nz)*ny*nz + (ny-1)*nz + gid%nz;
		idx2 = (gid/nz)*ny*nz + (ny-2)*nz + gid%nz;
		gid2 = gid + nx*nz;

		h0_new = h0[idx1] - 0.5 * pe1[gid];
		h1_new = h1[idx1] + 0.5 * pe0[gid];
		h0[idx1] = h0_new;
		h1[idx1] = h1_new;

		ph0[gid] -= cb1 * h0_new;
		ph1[gid] -= cb1 * h1_new;
		ph0[gid2] -= cb2 * h0[idx2];
		ph1[gid2] -= cb2 * h1[idx2];

		pe0[gid] = ca0 * pe0[gid] - cb0 * e0[idx2];
		pe1[gid] = ca0 * pe1[gid] - cb0 * e1[idx2];

		gid += get_global_size(0);
	}
} 


