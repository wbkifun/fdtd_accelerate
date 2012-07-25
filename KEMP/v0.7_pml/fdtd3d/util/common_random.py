import numpy as np
import unittest


randint = np.random.randint


def generate_ehs(nx, ny, nz, dtype, ufunc=''):
    ns = (nx, ny, nz)

    if ufunc == 'e':
        ehs = [np.zeros(ns, dtype=dtype) for i in range(3)] + \
                [np.random.rand(*ns).astype(dtype) for i in range(3)]

    elif ufunc == 'h':
        ehs = [np.random.rand(*ns).astype(dtype) for i in range(3)] + \
                [np.zeros(ns, dtype=dtype) for i in range(3)]

    elif ufunc == '':
        ehs = [np.random.rand(*ns).astype(dtype) for i in range(6)]

    return ehs



def generate_cs(nx, ny, nz, dtype, coeff_use):
    ns = (nx, ny, nz)

    if 'e' in coeff_use:
        ces = [np.random.rand(*ns).astype(dtype) for i in range(3)]
    else:
        ces = [np.ones(ns, dtype=dtype)*0.5 for i in range(3)]

    if 'h' in coeff_use:
        chs = [np.random.rand(*ns).astype(dtype) for i in range(3)]
    else:
        chs = [np.ones(ns, dtype=dtype)*0.5 for i in range(3)]

    return (ces, chs)



def sorted_two_ints(nx):
    p0 = p1 = 0
    while p0 == p1:
        p0, p1 = np.sort( randint(0, nx, 2) )

    return (p0, p1)



def set_two_points(shape, nx, ny, nz, iter=3):
    """
    Return the two tuples made of random integers
    """

    ns = (nx, ny, nz)
    pts_list = []

    if shape == 'point':
        for itr in xrange(iter):
            pt0, pt1 = [0, 0, 0], [0, 0, 0]

            for i, n in enumerate(ns):
                pt0[i] = pt1[i] = randint(0, n)

            pts_list.append( (pt0, pt1) )

    elif shape == 'line':
        for i, j, k in [(0,1,2), (2,0,1), (1,2,0)]:
            pt0, pt1 = [0, 0, 0], [0, 0, 0]

            pt0[i] = pt1[i] = randint(0, ns[i])
            pt0[j] = pt1[j] = randint(0, ns[j])
            pt0[k], pt1[k] = 0, ns[k]-1

            pts_list.append( (pt0, pt1) )

            for itr in xrange(iter):
                pt0, pt1 = [0, 0, 0], [0, 0, 0]

                pt0[i] = pt1[i] = randint(0, ns[i])
                pt0[j] = pt1[j] = randint(0, ns[j])
                pt0[k], pt1[k] = sorted_two_ints(ns[k])

                pts_list.append( (pt0, pt1) )

    elif shape == 'plane':
        for i, j, k in [(0,1,2), (2,0,1), (1,2,0)]:
            pt0, pt1 = [0, 0, 0], [0, 0, 0]

            pt0[i] = pt1[i] = randint(0, ns[i])
            pt0[j], pt1[j] = 0, ns[j]-1
            pt0[k], pt1[k] = 0, ns[k]-1

            pts_list.append( (pt0, pt1) )

            for itr in xrange(iter):
                pt0, pt1 = [0, 0, 0], [0, 0, 0]

                pt0[i] = pt1[i] = randint(0, ns[i])
                pt0[j], pt1[j] = sorted_two_ints(ns[j])
                pt0[k], pt1[k] = sorted_two_ints(ns[k])

                pts_list.append( (pt0, pt1) )

    elif shape == 'volume':
        pt0, pt1 = [0, 0, 0], [0, 0, 0]

        pt0[0], pt1[0] = 0, nx-1
        pt0[1], pt1[1] = 0, ny-1
        pt0[2], pt1[2] = 0, nz-1

        pts_list.append( (pt0, pt1) )

        for itr in xrange(iter):
            pt0, pt1 = [0, 0, 0], [0, 0, 0]

            pt0[0], pt1[0] = sorted_two_ints(nx)
            pt0[1], pt1[1] = sorted_two_ints(ny)
            pt0[2], pt1[2] = sorted_two_ints(nz)

            pts_list.append( (pt0, pt1) )

    return pts_list




class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.ns = (40, 50, 64)


    def test_sorted_two_ints(self):
        for itr in xrange(1000):
            p0, p1 = sorted_two_ints(self.ns[0])
            self.assertTrue(p0 < p1)


    def test_set_point(self):
        pts_list = set_two_points('point', *self.ns)
        for pt0, pt1 in pts_list:
            for p0, p1 in zip(pt0, pt1):
                self.assertEqual(p0, p1)
        self.assertNotEqual(pts_list[0], pts_list[1])


    def test_set_line(self):
        pts_list = set_two_points('line', *self.ns)
        matched = [0, 0, 0]
        for pt0, pt1 in pts_list:
            diff = [0, 0, 0]
            for i, p0, p1 in zip([0,1,2], pt0, pt1):
                if p0 != p1:
                    diff[i] = 1
            self.assertEqual(sum(diff), 1)
            matched[diff.index(1)] += 1
        self.assertEqual(matched[0], matched[1])
        self.assertEqual(matched[1], matched[2])
        self.assertNotEqual(pts_list[0], pts_list[1])


    def test_set_plane(self):
        pts_list = set_two_points('plane', *self.ns)
        matched = [0, 0, 0]
        for pt0, pt1 in pts_list:
            diff = [0, 0, 0]
            for i, p0, p1 in zip([0,1,2], pt0, pt1):
                if p0 != p1:
                    diff[i] = 1
            self.assertEqual(sum(diff), 2)
            matched[diff.index(0)] += 1
        self.assertEqual(matched[0], matched[1])
        self.assertEqual(matched[1], matched[2])
        self.assertNotEqual(pts_list[0], pts_list[1])


    def test_set_volume(self):
        pts_list = set_two_points('volume', *self.ns)
        for pt0, pt1 in pts_list:
            for p0, p1 in zip(pt0, pt1):
                self.assertNotEqual(p0, p1)
        self.assertNotEqual(pts_list[0], pts_list[1])



if __name__ == '__main__':
    unittest.main()
