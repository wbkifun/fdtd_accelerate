import doctest
import numpy as np
import unittest


def check_type(arg_name, arg, arg_type, element_type=None):
    """
    Check the type of the argument
    If the type is mismatch, the TypeError exception is raised.

    When the 'arg's type is a list or a tuple, 
    each element's type is also checked.

    >>> check_type('arg_name', 2, int)
    >>> check_type('arg_name', 3.4, (int, float))
    >>> check_type('arg_name', 'xy', str)
    >>> check_type('arg_name', (1.2, 2.3), tuple, float)
    >>> check_type('arg_name', [1.2, 2.3], list, float)
    >>> check_type('arg_name', ['a', 'b'], (list, tuple), str)
    """

    if not isinstance(arg, arg_type):
        raise TypeError("argument '%s' type must be a %s : %s is given" % \
                (arg_name, repr(arg_type), type(arg)) )

    if isinstance(arg, (list, tuple)):
        if element_type == None:
            raise TypeError( \
                "\n\tWhen the 'arg's type is a list or a tuple, \
                \n\targumnet 'element_type' must be specified." )

        for element in arg:
            if not isinstance(element, element_type):
                raise TypeError("argument '%s's element type must be a %s : %s is given" % \
                        (arg_name, repr(element_type), type(element)) )



def check_value(arg_name, arg, value):
    """
    Check if the argument is one of the values
    If the value is mismatch, the ValueError exception is raised.

    >>> check_value('arg_name', 'a', ('a', 'b', 'ab'))
    """

    if not arg in convert_to_tuple(value):
        repr_val = repr(value)
        if isinstance(value, (list, tuple)) and len(repr_val) > 40:
            repr_val = str(value[:2] + ['...'] + value[-2:]).replace("'", '')

        raise ValueError("argument '%s' value must be one of %s : %s is given" % \
                (arg_name, repr_val, repr(arg)) )



def binary_prefix_nbytes(nbytes):
    """
    Return a (converted nbytes, binary prefix) pair for the nbytes

    >>> binary_prefix_nbytes(2e9)
    (1.862645149230957, 'GiB')
    >>> binary_prefix_nbytes(2e6)
    (1.9073486328125, 'MiB')
    >>> binary_prefix_nbytes(2e3)
    (1.953125, 'KiB')
    >>> binary_prefix_nbytes(2)
    (2, 'Bytes')
    """

    check_type('nbytes', nbytes, (int, float))

    if nbytes >= 1024**3: 
        value = float(nbytes)/(1024**3)
        prefix_str = 'GiB'

    elif nbytes >= 1024**2: 
        value = float(nbytes)/(1024**2)
        prefix_str = 'MiB'

    elif nbytes >= 1024: 
        value = float(nbytes)/1024
        prefix_str = 'KiB'

    else:
        value = nbytes
        prefix_str = 'Bytes'

    return value, prefix_str



def replace_template_code(code, old_list, new_list):
    """
    Replace the macros in the template code

    >>> code = '''AA, BB'''
    >>> replace_template_code(code, ['AA', 'BB'], ['aa', str(22)])
    'aa, 22'
    """

    check_type('code', code, str)
    check_type('old_list', old_list, (list, tuple), str)
    check_type('new_list', new_list, (list, tuple), str)
    assert len(old_list) == len(new_list), \
            "arguments 'old_list' and 'new_list' do not have same length"

    for old, new in zip(old_list, new_list):
        code = code.replace(old, new)

    return code



def slices_two_points(pt0, pt1):
    """
    Return the tuple of slice indices from two points

    >>> slices_two_points((0, 0, 0), (10, 11, 12))
    (slice(0, 11, None), slice(0, 12, None), slice(0, 13, None))
    >>> slices_two_points((0, 0, 0), (10, 0, 12))
    (slice(0, 11, None), 0, slice(0, 13, None))
    >>> slices_two_points((1, 2, 3), (1, 2, 3))
    (1, 2, 3)
    """

    check_type('pt0', pt0, (list, tuple), int)
    check_type('pt1', pt1, (list, tuple), int)

    slices = [slice(p0, p1+1) if p0 != p1 else p0 for p0, p1 in zip(pt0, pt1)]

    return tuple(slices)



def two_points_slices(ns, slices):
    """
    Return the two points from the tuple of slice indices

    >>> two_points_slices((11, 12, 13), (slice(None, None), slice(None, None), slice(None, None)))
    ((0, 0, 0), (10, 11, 12))
    >>> two_points_slices((11, 12, 13), (slice(None, None), 0, slice(None, None)))
    ((0, 0, 0), (10, 0, 12))
    >>> two_points_slices((11, 12, 13), (1, 2, 3))
    ((1, 2, 3), (1, 2, 3))
    """

    check_type('slices', slices, (list, tuple), (slice, int))
    pt0 = [sl.indices(n)[0] if isinstance(sl, slice) else sl for n, sl in zip(ns, slices)]
    pt1 = [sl.indices(n)[1]-1  if isinstance(sl, slice) else sl for n, sl in zip(ns, slices)]

    return (tuple(pt0), tuple(pt1))



def shape_two_points(pt0, pt1, mul_x=1, is_dummy=False):
    """
    Return the shape from two points

    >>> shape_two_points((0, 0, 0), (10, 11, 12))
    (11, 12, 13)
    >>> shape_two_points((0, 0, 0), (10, 0, 12))
    (11, 13)
    >>> shape_two_points((0, 0, 0), (0, 0, 12))
    (13,)
    >>> shape_two_points((1, 2, 3), (1, 2, 3))
    (1,)
    >>> shape_two_points((0, 0, 0), (10, 11, 12), 2)
    (22, 12, 13)
    >>> shape_two_points((0, 0, 0), (10, 0, 12), 3)
    (33, 13)
    >>> shape_two_points((0, 0, 0), (0, 0, 12), 4)
    (52,)
    >>> shape_two_points((1, 2, 3), (1, 2, 3), 2)
    (2,)
    >>> shape_two_points((0, 0, 0), (0, 0, 12), is_dummy=True)
    (1, 1, 13)
    >>> shape_two_points((1, 2, 3), (1, 2, 3), is_dummy=True)
    (1, 1, 1)
    """

    check_type('pt0', pt0, (list, tuple), int)
    check_type('pt1', pt1, (list, tuple), int)
    check_type('mul_x', mul_x, int)

    shape = []
    for p0, p1 in zip(pt0, pt1):
        value = abs(p1 - p0) + 1
        if value == 1:
            if is_dummy:
                shape.append(value)
        else:
            shape.append(value)

    if shape == []:
        return (mul_x,)
    else:
        shape[0] *= mul_x
        return tuple(shape)



def convert_to_tuple(arg):
    """
    Return the tuple which is converted from the arbitrary argument

    >>> convert_to_tuple(3)
    (3,)
    >>> convert_to_tuple(['a', 'b'])
    ('a', 'b')
    """

    if isinstance(arg, (list, tuple)):
        return tuple(arg)
    else:
        return (arg,)



def convert_index(n, num):
    """
    Return the index which is converted to positive integer

    >>> convert_index(10, 3)
    3
    >>> convert_index(10, -1)
    9
    >>> convert_index(10, 0.5)
    5
    """

    check_type('n', n, (int, np.int32))
    check_type('num', num, (int, float))

    if isinstance(num, int):
        if num >= 0:
            return num
        else:
            return n + num

    elif isinstance(num, float):
        return int(n * num)



def convert_indices(ns, pt0):
    """
    Return the indices which are converted to positive integers

    >>> convert_indices((10, 12, 14), (1, 2, 3))
    (1, 2, 3)
    >>> convert_indices((10, 12, 14), (-1, -2, -3))
    (9, 10, 11)
    >>> convert_indices((10, 12, 14), (0.1, 0.2, 0.3))
    (1, 2, 4)
    """

    check_type('ns', ns, (list, tuple), (int, np.int32))
    check_type('pt0', pt0, (list, tuple), (int, float))

    pt = [convert_index(n, p0) for n, p0 in zip(ns, pt0)]

    return tuple(pt)



def overlap_two_slices(ns, slices0, slices1):
    """
    Return the slice which is overlapped slices

    >>> ns = (10, 20, 30)
    >>> slices0 = (slice(-2,None), slice(None,None), slice(None,None))
    >>> slices1 = (slice(None,None), slice(None,None), slice(None,None))
    >>> overlap_two_slices(ns, slices0, slices1)
    (slice(8, 10, None), slice(0, 20, None), slice(0, 30, None))
    >>> slices1 = (slice(-4,-2), slice(None,None), slice(None,None))
    >>> overlap_two_slices(ns, slices0, slices1)

    >>> slices0 = (slice(5, 6), slice(7, 12), slice(12, 17))
    >>> slices1 = (slice(5, 6), slice(7, 12), slice(12, 17))
    >>> overlap_two_slices(ns, slices0, slices1)
    (5, slice(7, 12, None), slice(12, 17, None))
    """

    check_type('ns', ns, (list, tuple), int)
    check_type('slices0', slices0, (list, tuple), (slice, int))
    check_type('slices1', slices1, (list, tuple), (slice, int))

    assert len(ns) == len(slices0) == len(slices1), \
            'The argument lists must have same length. %s, %s, %s' % \
            (len(ns), len(slices0), len(slices1))

    slices = []
    for n, sl0, sl1 in zip(ns, slices0, slices1):
        if isinstance(sl0, int):
            sl0 = slice(sl0, sl0+1)
        if isinstance(sl1, int):
            sl1 = slice(sl1, sl1+1)

        set0 = set( range(*sl0.indices(n)) )
        set1 = set( range(*sl1.indices(n)) )
        overlap = sorted( list( set0.intersection(set1) ) )

        if len(overlap) > 0: 
            if overlap[0] == overlap[-1]:
                slices.append( overlap[0] )
            else:
                slices.append( slice(overlap[0], overlap[-1]+1) )

    if len(slices) == len(slices0):
        return tuple(slices)
    else:
        return None



def overlap_two_lines(line0, line1):
    """
    Return the two lines which is overlapped

    >>> overlap_two_lines((8, 9), (0, 9))
    (8, 9)
    >>> overlap_two_lines((0, 1), (1, 2))
    (1, 1)
    >>> overlap_two_lines((0, 1), (2, 3))

    >>> overlap_two_lines((-1, 9), (-1, 1))
    (-1, 1)
    >>> overlap_two_lines((0, 10), (8, 10))
    (8, 10)
    """

    check_type('line0', line0, (list, tuple), int)
    check_type('line1', line1, (list, tuple), int)

    x0, x1 = line0
    x2, x3 = line1

    set0 = set( range(x0, x1+1) )
    set1 = set( range(x2, x3+1) )
    overlap = sorted( list( set0.intersection(set1) ) )

    if len(overlap) > 0: 
        return (overlap[0], overlap[-1])
    else:
        return None



def overlap_two_regions(pt0, pt1, pt2, pt3):
    """
    Return the two points(tuple) which is overlapped regions

    >>> overlap_two_regions((8,0,0), (9,19,29), (0,0,0), (9,19,29))
    ((8, 0, 0), (9, 19, 29))
    >>> overlap_two_regions((0,0,0), (1,19,29), (1,0,0), (2,19,29))
    ((1, 0, 0), (1, 19, 29))
    >>> overlap_two_regions((0,0,0), (1,19,29), (2,0,0), (3,19,29))

    """

    check_type('pt0', pt0, (list, tuple), int)
    check_type('pt1', pt1, (list, tuple), int)
    check_type('pt2', pt2, (list, tuple), int)
    check_type('pt3', pt3, (list, tuple), int)

    assert len(pt0) == len(pt1) == len(pt2) == len(pt3), \
            'The points must have same length.'

    pt4, pt5 = [], []
    for p0, p1, p2, p3 in zip(pt0, pt1, pt2, pt3):
        overlap = overlap_two_lines((p0, p1), (p2, p3))
        if overlap != None:
            pt4.append(overlap[0])
            pt5.append(overlap[-1])

    if len(pt4) == len(pt5) == len(pt0):
        return (tuple(pt4), tuple(pt5))
    else:
        return None



def append_instance(instance_list, instance):
    priority_dict = { \
                'core':0, 'current':1, \
                'pml':2, 'incident':3, \
                'pbc':4, 'mpi':5, 'exchange':6}

    new = priority_dict[instance.priority_type]
    index = len(instance_list)
    for i, inst in enumerate(instance_list):
        old = priority_dict[inst.priority_type]
        if new < old:
            index = i
            break

    instance_list.insert(index, instance)




class TestFunctions(unittest.TestCase):
    def test_doctest(self):
        doctest.testmod()


    def test_check_type(self):
        self.assertRaises(TypeError, check_type, '', 2, float)
        self.assertRaises(TypeError, check_type, '', 3.2, str)
        self.assertRaises(TypeError, check_type, '', 3.4, (int, str))
        self.assertRaises(TypeError, check_type, '', [1, 2], list)
        self.assertRaises(TypeError, check_type, '', (1.2, 2.3), tuple, int)
        self.assertRaises(TypeError, check_type, '', ['a', 'b'], tuple, int)
        self.assertRaises(TypeError, check_type, '', ['a', 'b', {'c':3}], list, str)


    def test_check_value(self):
        self.assertRaises(ValueError, check_value, '', 'a', ('b', 'c'))


    def test_replace_template_code(self):
        self.assertRaises(AssertionError, replace_template_code, \
                'AA, BB', ['AA', 'BB'], ['a', 'b', 'c'])



if __name__ == '__main__':
    unittest.main()
