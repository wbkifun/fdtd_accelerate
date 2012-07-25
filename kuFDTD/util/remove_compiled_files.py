#!/usr/bin/env python

from util_functions import *

def remove_compiled_files(dir, file):
    fpath = dir + '/' + file
    os.remove(fpath)


if __name__ == '__main__':
    import sys
    try:
        mydir = sys.argv[1]
    except IndexError:
        mydir = '.'

    suffix_list = ['.o', '.so', '.pyc']
    function = remove_compiled_files

    matched_file_count = \
            find_recursively_matched_files( \
            mydir, suffix_list, function, False)

    if matched_file_count == 0:
        print 'No matched files.'
    else:
        opt = raw_input('These files are removed!\nDo you really? [Y/n]: ')
        if opt == '' or opt == 'y' or opt == 'Y':
            find_recursively_matched_files( \
                    mydir, suffix_list, function, True)
            print 'Done!'
        else:
            print 'Canceled!'
