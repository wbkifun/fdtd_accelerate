import doctest
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
        raise ValueError("argument '%s' value must be one of %s : %s is given" % \
                (arg_name, repr(value), repr(arg)) )



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



def slice_index_two_points(pt0, pt1):
    """
    Return the tuple of slice indices from two points

    >>> slice_index_two_points((0, 0, 0), (10, 11, 12))
    (slice(0, 11, None), slice(0, 12, None), slice(0, 13, None))
    >>> slice_index_two_points((0, 0, 0), (10, 0, 12))
    (slice(0, 11, None), 0, slice(0, 13, None))
    >>> slice_index_two_points((1, 2, 3), (1, 2, 3))
    (1, 2, 3)
    """

    check_type('pt0', pt0, (list, tuple), int)
    check_type('pt1', pt1, (list, tuple), int)

    slidx = []
    for p0, p1 in zip(pt0, pt1):
        if p0 == p1:
            slidx.append(p0)
        else:
            slidx.append(slice(p0, p1+1))

    return tuple(slidx)



def shape_two_points(pt0, pt1):
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
    """

    check_type('pt0', pt0, (list, tuple), int)
    check_type('pt1', pt1, (list, tuple), int)

    shape = []
    for p0, p1 in zip(pt0, pt1):
        if p0 != p1:
            shape.append( abs(p1 - p0) + 1 )

    if shape == []:
        return (1,)
    else:
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
