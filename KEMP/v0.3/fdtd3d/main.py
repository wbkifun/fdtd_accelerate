import h5py
import numpy as np
import os
import pickle
import sys
import types
from datetime import datetime
from time import sleep

from kemp.fdtd3d.util import common
from kemp.fdtd3d import cpu, node


class Fdtd3d:
    def __init__(self, geometry_h5_path, max_tstep, mpi_shape, pbc_axes='', target_device='all', precision_float='single', **kargs):
        """
        """

        common.check_type('geometry_h5_path', geometry_h5_path, str)
        common.check_type('max_tstep', max_tstep, int)
        common.check_type('mpi_shape', mpi_shape, (list, tuple), int)
        common.check_type('pbc_axes', pbc_axes, str)
        common.check_type('target_device', target_device, str)
        common.check_value('precision_float', precision_float, ['single', 'double'])

        # import modules
        global is_mpi, is_gpu

        is_mpi = False if mpi_shape == (1, 1, 1) else True

        if is_mpi:
            global network, common_mpi, comm, size, rank, coord
            from mpi4py import MPI
            from kemp.fdtd3d import network
            from kemp.fdtd3d.util import common_mpi
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            coord = common_mpi.my_coord(rank, mpi_shape)

        is_master = False if is_mpi and rank != 0 else True
        is_cpu = True if target_device == 'all' or 'cpu' in target_device else False
        is_gpu = True if target_device == 'all' or 'gpu' in target_device else False

        if is_mpi:
            if reduce(lambda a,b:a*b, mpi_shape) != size:
                if is_master:
                    print("The MPI size %d is not matched the mpi_shape %s" % (size, mpi_shape) )
                sys.exit()

        if is_gpu:
            try:
                global cl, gpu, common_gpu
                import pyopencl as cl
                from kemp.fdtd3d import gpu
                from kemp.fdtd3d.util import common_gpu
            except:
                if is_master:
                    print("The 'pyopencl' module is not found.")

                if is_cpu:
                    if is_master:
                        print("The CPU is only used.")
                    target_device = 'cpu'
                    is_gpu = False
                else:
                    sys.exit()

        # read from the h5 file
        try:
            h5f = h5py.File(geometry_h5_path, 'r')
            coeff_use = h5f.attrs['coeff_use']
            nx = h5f.attrs['nx']
            ny = h5f.attrs['ny']
            nz = h5f.attrs['nz']
        except:
            if is_master:
                print( repr(sys.exc_info()) )
                print("To load the geometry HDF5 file '%s' is failed." % geometry_h5_path)
            sys.exit()

        # local variables
        device_nx_list = kargs['device_nx_list'] if kargs.has_key('device_nx_list') else None
        ny_list = kargs['ny_list'] if kargs.has_key('ny_list') else None
        nz_list = kargs['nz_list'] if kargs.has_key('nz_list') else None

        # Set the number of device and the device_n_list
        ndev = 1 if is_cpu else 0
        if is_gpu:
            try:
                gpu_devices = common_gpu.gpu_device_list(print_info=False)
                context = cl.Context(gpu_devices)
                ndev += len(gpu_devices)
            except Exception as errinst:
                if is_master:
                    print( repr(sys.exc_info()) )
                    print("To get the GPU devices is failed. The CPU is only used.")
                target_device = 'cpu'
                is_gpu = False

        if is_mpi:
            mi, mj, mk = coord
            dnx_list = device_nx_list[mi*ndev:(mi+1)*ndev]
            dny = ny_list[mj]
            dnz = nz_list[mk]
        else:
            dnx_list = device_nx_list
            dny = ny_list[0]
            dnz = nz_list[0]
            
        total_ndev = mpi_shape[0] * ndev
        if len(device_nx_list) != total_ndev:
            if is_master:
                print("The device_nx_list %s is not matched with the number of total devices %d." % (device_nx_list, total_ndev) )
            sys.exit()

        # create the mainf_list and the buffer_dict
        buffer_dict = {}
        if is_mpi:
            # create BufferFields instances
            snx = sum(dnx_list) - ndev + 1
            sny, snz = dny, dnz

            mpi_target_dict = common_mpi.mpi_target_dict(rank, mpi_shape, pbc_axes)
            for direction, target_rank in mpi_target_dict.items():
                if target_rank != None:
                    n0, n1 = {'x': (sny, snz), 'y': (snx, snz), 'z': (snx, sny)}[direction[0]]
                    bufferf = cpu.BufferFields(direction, target_rank, n0, n1, coeff_use, precision_float)
                    buffer_dict[direction] = bufferf
                    #network.ExchangeMpi(bufferf, target_rank, max_tstep)
                    #network.ExchangeMpiNoSplitBlock(bufferf, target_rank)
                    #network.ExchangeMpiBlock(bufferf, target_rank)

        mainf_list = []
        if is_cpu:
            mainf_list += [cpu.Fields(dnx_list.pop(0), dny, dnz, coeff_use, precision_float, use_cpu_core=1)]

        if is_gpu:
            mainf_list += [gpu.Fields(context, gpu_device, dnx, dny, dnz, coeff_use, precision_float) for gpu_device, dnx in zip(gpu_devices, dnx_list)]

        # create node.Fields instance
        nodef = node.Fields(mainf_list, buffer_dict)

        # create nodePbc instance
        node_pbc_axes = ''.join([axis for i, axis in enumerate(['x', 'y', 'z']) if mpi_shape[i] == 1 and axis in pbc_axes])
        if node_pbc_axes != '':
            node.Pbc(nodef, node_pbc_axes)

        # create update instances
        node.Core(nodef)
        for bufferf in nodef.buffer_dict.values():
            #network.ExchangeMpiSplitBlock(bufferf)
            network.ExchangeMpiSplitNonBlock(bufferf, max_tstep)
            '''
            if rank == 0:
                direction = 'x+'
                target_rank = 1
            elif rank == 1:
                direction = 'x-'
                target_rank = 0

            #network.ExchangeMpiNoBufferBlock(nodef, target_rank, direction)    # no buffer, block
            self.mpi_instance_list = []
            self.mpi_instance_list.append( network.ExchangeMpiNoBufferNonBlock(nodef, target_rank, direction) )
            '''

        # accum_sub_ns_dict, node_pts
        if is_mpi:
            asn_dict = common_mpi.accum_sub_ns_dict(mpi_shape, ndev, device_nx_list, ny_list, nz_list)
            axes = ['x', 'y', 'z']
            node_pt0 = [asn_dict[ax][m] for ax, m in zip(axes, coord)]
            node_pt1 = [asn_dict[ax][m+1] - 1 for ax, m in zip(axes, coord)]

        # global variables
        self.max_tstep = max_tstep
        self.mpi_shape = mpi_shape
        #self.ns = (nx, ny, nz)
        self.ns = (asn_dict['x'][-1], asn_dict['y'][-1], asn_dict['z'][-1]) if is_mpi else nodef.ns

        self.nodef = nodef
        self.is_master = is_master

        if is_mpi:
            self.asn_dict = asn_dict
            self.node_pt0 = node_pt0
            self.node_pt1 = node_pt1

        # for savefields
        self.savef_tag_list = []
        self.savef_list = []


    def set_pml(self, axes='', depth=10, sigma=0, alpha=0, gamma=0, m_order=0):
        pass


    def set_incident_direct(self, str_f, pt0, pt1, tfunc, spatial_value=1., is_overwrite=False):
        common.check_value('str_f', str_f, ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'))
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))
        common.check_type('tfunc', tfunc, types.FunctionType)
        common.check_type('spatial_value', spatial_value, \
                (np.ndarray, np.number, types.FloatType, types.IntType) )
        common.check_type('is_overwrite', is_overwrite, bool)

        pt0 = list( common.convert_indices(self.ns, pt0) )
        pt1 = list( common.convert_indices(self.ns, pt1) )

        if is_mpi:
            node_pt0 = list(self.node_pt0)
            node_pt1 = list(self.node_pt1)

            for i, axis in enumerate(['x', 'y', 'z']):
                if self.nodef.buffer_dict.has_key('%s+' % axis):
                    node_pt1[i] += 1
                if self.nodef.buffer_dict.has_key('%s-' % axis):
                    node_pt0[i] -= 1

                if coord[i] == 0 and pt0[i] == 0:
                    pt0[i] -= 1
                if coord[i] == self.mpi_shape[i]-1 and pt1[i] == self.ns[i]-1:
                    pt1[i] += 1

            overlap = common.overlap_two_regions(node_pt0, node_pt1, pt0, pt1)
            if overlap != None:
                sx0, sy0, sz0 = self.node_pt0
                ox0, oy0, oz0 = overlap[0]
                ox1, oy1, oz1 = overlap[1]

                local_pt0 = (ox0-sx0, oy0-sy0, oz0-sz0)
                local_pt1 = (ox1-sx0, oy1-sy0, oz1-sz0)

                node.IncidentDirect(self.nodef, str_f, local_pt0, local_pt1, tfunc, spatial_value, is_overwrite)

        else:
            node.IncidentDirect(self.nodef, str_f, pt0, pt1, tfunc, spatial_value, is_overwrite)


    def set_savefields(self, str_f, pt0, pt1, tstep_range, dir_path, tag=0):
        common.check_type('str_f', str_f, (str, list, tuple), str)
        common.check_type('pt0', pt0, (list, tuple), (int, float))
        common.check_type('pt1', pt1, (list, tuple), (int, float))
        common.check_type('tstep_range', tstep_range, (list, tuple), int)
        common.check_type('dir_path', dir_path, str)
        common.check_type('tag', tag, int)

        pt0 = common.convert_indices(self.ns, pt0)
        pt1 = common.convert_indices(self.ns, pt1)
        t0 = common.convert_index(self.max_tstep, tstep_range[0])
        t1 = common.convert_index(self.max_tstep, tstep_range[1]) + 1
        tgap = tstep_range[2]
        tmax = self.max_tstep

        tag = tag if tag not in self.savef_tag_list else max(self.savef_tag_list)+1
        self.savef_tag_list.append(tag)

        if is_mpi:
            overlap = common.overlap_two_regions(self.node_pt0, self.node_pt1, pt0, pt1)
            if overlap != None:
                sx0, sy0, sz0 = self.node_pt0
                ox0, oy0, oz0 = overlap[0]
                ox1, oy1, oz1 = overlap[1]

                local_pt0 = (ox0-sx0, oy0-sy0, oz0-sz0)
                local_pt1 = (ox1-sx0, oy1-sy0, oz1-sz0)

                savef = node.SaveFields(self.nodef, str_f, local_pt0, local_pt1, dir_path, tag, tmax, True, rank)
                self.savef_list.append(savef)
        else:
            savef = node.SaveFields(self.nodef, str_f, pt0, pt1, dir_path, tag, tmax)
            self.savef_list.append(savef)

        # save the subdomain informations as pickle
        if self.is_master:
            if is_mpi:
                divide_info_dict = common_mpi.divide_info_dict(size, self.mpi_shape, pt0, pt1, self.asn_dict)
            else:
                divide_info_dict = { \
                        'shape': common.shape_two_points(pt0, pt1), \
                        'pt0': pt0, \
                        'pt1': pt1, \
                        'anx_list': self.nodef.accum_nx_list }

            divide_info_dict['str_fs'] = common.convert_to_tuple(str_f)
            divide_info_dict['tmax'] = tmax
            divide_info_dict['tgap'] = tgap
            pkl_file = open(dir_path + 'divide_info.pkl', 'wb')
            pickle.dump(divide_info_dict, pkl_file)
            pkl_file.close()

        self.t0 = t0
        self.t1 = t1
        self.tgap = tgap


    def run_timeloop(self):
        tmax = self.max_tstep

        # print time stamp
        if self.is_master:
            t0 = datetime.now()

        gtmp = node.GetFields(self.nodef, 'ez', (0, 0, 0), (0, 0, 0))
        # main time-loop
        for tstep in xrange(1, tmax+1):
            self.nodef.update_e()
            self.nodef.update_h()

            '''
            gtmp.wait()
            if tstep % 10 == 0 and self.is_master:
                print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
                sys.stdout.flush()

            for savef in self.savef_list:
                if tstep % self.tgap == 0 and self.t0 <= tstep <= self.t1:
                    savef.save_fields(tstep)
            '''

        # finalize
        #self.nodef.cpuf.enqueue_barrier()
        gtmp.wait()
        if self.is_master: 
            print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
            print('')
        sleep(0.5)
