import numpy as np
import h5py as h5

from kemp.fdtd3d.util import common

from fields import Fields
from get_fields import GetFields


class SaveFields:
    def __init__(self, node_fields, str_f, pt0, pt1, dir_path, tag, max_tstep, is_mpi=False, rank=0):
        """
        """

        common.check_type('node_fields', node_fields, Fields)
        common.check_type('str_f', str_f, (str, list, tuple), str)
        common.check_type('pt0', pt0, (list, tuple), int)
        common.check_type('pt1', pt1, (list, tuple), int)
        common.check_type('dir_path', dir_path, str)
        common.check_type('tag', tag, int)
        common.check_type('max_tstep', max_tstep, int)

        # local variables
        nodef = node_fields
        str_fs = common.convert_to_tuple(str_f)

        # setup
        getf = GetFields(nodef, str_fs, pt0, pt1)
        ndigit = int( ('%e' % max_tstep).split('+')[1] ) + 1
        fpath_form = dir_path + '%%.%dd_tag%d.h5' % (ndigit, tag)
        if is_mpi:
            fpath_form = fpath_form.rstrip('.h5') + '_mpi%d.h5' % rank

        # global variables
        self.str_fs = str_fs
        self.pt0 = pt0
        self.pt0 = pt1
        self.getf = getf
        self.fpath_form = fpath_form


    def save_fields(self, tstep):
        fpath = self.fpath_form % tstep
        h5f = h5.File(fpath, 'w')
        self.getf.wait()
        for str_f in self.str_fs:
            h5f.create_dataset(str_f, data=self.getf.get_fields(str_f), compression='gzip')
        h5f.close()
