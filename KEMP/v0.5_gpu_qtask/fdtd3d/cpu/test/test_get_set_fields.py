import numpy as np
import unittest

import sys, os
sys.path.append( os.path.expanduser('~') )
from kemp.fdtd3d.util import common, common_random
from kemp.fdtd3d.cpu import GetFields, SetFields, Fields


class TestGetFields(unittest.TestCase):
    def __init__(self, args):
        super(TestGetFields, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1 = self.args

        slice_xyz = common.slices_two_points(pt0, pt1)
        str_fs = common.convert_to_tuple(str_f)

        # instance
        fields = Fields(nx, ny, nz, '', 'single')
        getf = GetFields(fields, str_f, pt0, pt1) 
        
        # host allocations
        ehs = common_random.generate_ehs(nx, ny, nz, fields.dtype)
        eh_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], ehs) )
        fields.set_ehs(*ehs)

        # verify
        getf.get_event().wait()

        for str_f in str_fs:
            original = eh_dict[str_f][slice_xyz]
            copy = getf.get_fields(str_f)
            norm = np.linalg.norm(original - copy)
            self.assertEqual(norm, 0, '%s, %g' % (self.args, norm))



class TestSetFields(unittest.TestCase):
    def __init__(self, args):
        super(TestSetFields, self).__init__() 
        self.args = args


    def runTest(self):
        nx, ny, nz, str_f, pt0, pt1, is_array = self.args

        slice_xyz = common.slices_two_points(pt0, pt1)
        str_fs = common.convert_to_tuple(str_f)

        # instance
        fields = Fields(nx, ny, nz, '', 'single')
        setf = SetFields(fields, str_f, pt0, pt1, is_array) 
        
        # generate random source
        if is_array:
            shape = common.shape_two_points(pt0, pt1, len(str_fs))
            value = np.random.rand(*shape).astype(fields.dtype)
            split_value = np.split(value, len(str_fs))
            split_value_dict = dict( zip(str_fs, split_value) )
        else:
            value = np.random.ranf()

        # host allocations
        ehs = [np.zeros(fields.ns, dtype=fields.dtype) for i in range(6)]
        eh_dict = dict( zip(['ex', 'ey', 'ez', 'hx', 'hy', 'hz'], ehs) )

        # verify
        for str_f in str_fs:
            if is_array:
                eh_dict[str_f][slice_xyz] = split_value_dict[str_f]
            else:
                eh_dict[str_f][slice_xyz] = value

        setf.set_fields(value)
        fields.enqueue_barrier()

        for str_f in str_fs:
            original = eh_dict[str_f]
            copy = fields.get(str_f)[:,:,fields.slice_z]
            norm = np.linalg.norm(original - copy)
            self.assertEqual(norm, 0, '%s, %g' % (self.args, norm))



if __name__ == '__main__':
    ns = nx, ny, nz = 40, 50, 60
    suite = unittest.TestSuite() 

    # Test GetFieds
    # single set
    suite.addTest(TestGetFields( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1)) ))
    suite.addTest(TestGetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (0, 1, nz-1)) ))

    # boundary exchange
    args_list1 = [(nx, ny, nz, str_f, pt0, pt1) \
            for str_f, pt0, pt1 in [
                    [('ey', 'ez'), (0, 0, 0), (0, ny-2, nz-2)], 
                    [('ex', 'ez'), (0, 0, 0), (nx-2, 0, nz-2)],
                    [('ex', 'ey'), (0, 0, 0), (nx-2, ny-2, 0)],
                    [('hy', 'hz'), (nx-1, 1, 1), (nx-1, ny-1, nz-1)],
                    [('hx', 'hz'), (1, ny-1, 1), (nx-1, ny-1, nz-1)],
                    [('hx', 'hy'), (1, 1, nz-1), (nx-1, ny-1, nz-1)] ]]
    suite.addTests(TestGetFields(args) for args in args_list1)

    # random sets
    args_list2 = [(nx, ny, nz, str_f, pt0, pt1) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, *ns) ]
    suite.addTests(TestGetFields(args) for args in args_list2) 


    # Test SetFieds
    # single sets
    suite.addTest(TestSetFields( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), False) ))
    suite.addTest(TestSetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (0, 1, nz-1), False) ))
    suite.addTest(TestSetFields( (nx, ny, nz, 'ex', (0, 1, 0), (0, 1, nz-1), True) ))
    suite.addTest(TestSetFields( (nx, ny, nz, ['ex', 'ey'], (0, 1, 0), (0, 1, nz-1), True) ))

    # boundary exchange
    args_list3 = [(nx, ny, nz, str_f, pt0, pt1, is_array) \
            for str_f, pt0, pt1, is_array in [
                    [('ey', 'ez'), (0, 0, 0), (0, ny-2, nz-2), True], 
                    [('ex', 'ez'), (0, 0, 0), (nx-2, 0, nz-2), True],
                    [('ex', 'ey'), (0, 0, 0), (nx-2, ny-2, 0), True],
                    [('hy', 'hz'), (nx-1, 1, 1), (nx-1, ny-1, nz-1), True],
                    [('hx', 'hz'), (1, ny-1, 1), (nx-1, ny-1, nz-1), True],
                    [('hx', 'hy'), (1, 1, nz-1), (nx-1, ny-1, nz-1), True] ]]
    suite.addTests(TestSetFields(args) for args in args_list3)

    # random sets
    args_list4 = [(nx, ny, nz, str_f, pt0, pt1, is_array) \
            for str_f in ['ex', 'ey', 'ez', 'hx', 'hy', 'hz'] \
            for shape in ['point', 'line', 'plane', 'volume'] \
            for pt0, pt1 in common_random.set_two_points(shape, *ns) \
            for is_array in [False, True] ]
    suite.addTests(TestSetFields(args) for args in args_list4) 

    unittest.TextTestRunner().run(suite) 
