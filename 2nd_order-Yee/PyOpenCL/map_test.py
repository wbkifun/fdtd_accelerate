#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import numpy.linalg as la


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

#ary = np.arange(3*4*5).reshape(3,4,5, order="C").astype(np.float32)
ary = np.arange(3*4*5).reshape(3,4,5, order="F").astype(np.float32)
#ary = np.arange(3*4*5).astype(np.float32)

mf = cl.mem_flags
flags = mf.READ_WRITE | mf.COPY_HOST_PTR | mf.ALLOC_HOST_PTR
buf = cl.Buffer(ctx, flags, hostbuf = ary)
queue.finish()

ar2 = np.empty_like(ary)
cl.enqueue_read_buffer(queue, buf, ar2)
print la.norm(ary-ar2), ary.strides, ar2.strides

#ar3, evt = cl.enqueue_map_buffer(queue, buf, cl.map_flags.READ, 0, ary.shape, ary.dtype, "C")
ar3, evt = cl.enqueue_map_buffer(queue, buf, cl.map_flags.READ, 0, ary.shape, ary.dtype, "F")
print la.norm(ary-ar3), ary.strides, ar3.strides
