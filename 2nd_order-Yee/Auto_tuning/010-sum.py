#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import numpy.linalg as la

kernels = """
__kernel void sum(__global const float *a, __global const float *b, __global float *c) {
	int gid = get_global_id(0);
	c[gid] = a[gid] + b[gid];
}
"""

n = 50000000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

prg = cl.Program(ctx, kernels).build()
print 'aaa'
for i in xrange(10000):
	prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)
print 'bbb'
'''
a_plus_b = np.empty_like(a)
cl.enqueue_read_buffer(queue, dest_buf, a_plus_b).wait()

print la.norm(a_plus_b - (a+b))
'''
