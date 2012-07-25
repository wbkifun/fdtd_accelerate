#!/usr/bin/env python

from kemp import Fdtd, Source, SaveField


fdtd = Fdtd((x0,y0,z0), (x1,y1,z1), dx=(arr,arr,arr), tmax=)
fdtd = Fdtd((arr,arr,arr), tmax=)
fdtd = Fdtd((nx, ny, nz), dx=, tmax=)

fdtd.set_geometry('./intersection.h5')
fdtd.set_boundary(pml='xy', pbc='z', pml_opt=(n,sigma,alpha,gamma, m))

# Source.Direct('field', (x0,y0,z0), (x1,y1,z1), tfunc, tfunc_args, sfunc=None, sfunc_args=None)
# Source.Tfsf((x0,y0,z0), (x1,y1,z1), tfunc, tfunc_args)	# The tstep is excluded from tfunc_args. The tstep is default option.
# fdtd.set_source(obj or [obj,...])
line_sin = Source.Direct('ez', (nx/2, ny/2, 0), (nx/2, ny/2, nz-1), np.sin, (omega, phase))	
fdtd.set_source(line_sin)


# SaveField('field', (x0,y0,z0), (x1,y1,z1), (t0, t1, step))
# fdtd.set_savefields(obj or [obj,...])
save_ez = SaveField('ez', (0, 0, nz/2), (-1, -1, nz/2), (None, None, 100))
save_hx = SaveField('hx', (0, 0, nz/2), (-1, -1, nz/2), (None, None, 100))
fdtd.set_savefields([save_ez, save_hx])

fdtd.prepare()
fdtd.run_timeloop()
