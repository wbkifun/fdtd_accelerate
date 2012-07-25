import numpy as np
import unittest

randint = np.random.randint


def sorted_two_ints(nx):
    p0 = p1 = 0
    while p0 == p1:
        p0, p1 = np.sort( randint(0, nx, 2) )

    return (p0, p1)



def random_set_two_points(shape, nx, ny, nz, iter=3):
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


    def test_random_set_point(self):
        pts_list = random_set_two_points('point', *self.ns)
        for pt0, pt1 in pts_list:
            for p0, p1 in zip(pt0, pt1):
                self.assertEqual(p0, p1)
        self.assertNotEqual(pts_list[0], pts_list[1])


    def test_random_set_line(self):
        pts_list = random_set_two_points('line', *self.ns)
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


    def test_random_set_plane(self):
        pts_list = random_set_two_points('plane', *self.ns)
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


    def test_random_set_volume(self):
        pts_list = random_set_two_points('volume', *self.ns)
        for pt0, pt1 in pts_list:
            for p0, p1 in zip(pt0, pt1):
                self.assertNotEqual(p0, p1)
        self.assertNotEqual(pts_list[0], pts_list[1])



if __name__ == '__main__':
    unittest.main()
