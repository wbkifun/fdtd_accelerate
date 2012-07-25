from kemp import Fdtd3d

fdtd = Fdtd3d( \
        geometry_h5_path = 'test/geo.h5', \
        max_tstep = 1000, \
        mpi_shape = (1, 1, 1), \
        pbc_axes = '', \
        target_device = 'gpu', \
        precision_float = 'single', \
        #device_nx_list = [96, 96, 96, 96, 96, 96], \
        device_nx_list = [160], \
        ny_list = [960], \
        nz_list = [960] )

tf = lambda tstep: 50*np.sin(0.05 * tstep)
fdtd.set_incident_direct( \
        str_f = 'ez', \
        pt0 = (180, 20, 0), pt1 = (180, 20, -1), \
        tfunc = tf)

fdtd.set_incident_direct('ez', (340, 40, 0), (340, 40, -1), tf)
fdtd.set_incident_direct('ez', (620, 60, 0), (620, 60, -1), tf)

fdtd.set_savefields( \
        str_f = 'ez', \
        pt0 = (0, 0, 0.5), pt1 = (-1, -1, 0.5), \
        tstep_range = (1, -1, 10), \
        dir_path = 'test/h5/')

fdtd.run_timeloop()
