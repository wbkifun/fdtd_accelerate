#!/usr/bin/env python

from util_functions import *

def compile_c_files(dir, file):
	current_dir = os.getcwd()
	os.chdir(dir)
	o_file = file.replace('.c', '.o')
	so_file = file.replace('.c', '.so')

	print 'compile,link %s' %(dir+'/'+file)
	#cmd = 'gcc -O3 -fpic -g -I. -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -shared %s -o %s' %(file, so_file)
	#cmd = 'gcc -O3 -fpic -msse -fopenmp -I. -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -c %s -o %s' %(file, o_file)
	#cmd = 'gcc -O3 -fpic -msse -I. -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -shared %s -o %s' %(file, so_file)
	cmd = 'gcc -O3 -fpic -msse -pthread -I. -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -shared %s -o %s' %(file, so_file)
	os.system(cmd)
	#cmd = 'gcc -shared -o %s %s' %(so_file, o_file)
	#os.system(cmd)

	os.chdir(current_dir)
	# print dir.count('/') 
	# print '../'*dir.count('/') 
	# os.chdir( '../'*dir.count('/') )
	# os.chdir( 'util' )
	
	
if __name__ == '__main__':
	import sys
	try:
		mydir = sys.argv[1]
	except IndexError:
		mydir = '.'
		
	suffix_list = ['.c']
	function = compile_c_files
	
	matched_file_count = \
			find_recursively_matched_files( \
			mydir, suffix_list, function, False)
			
	if matched_file_count == 0:
		print 'No matched files.'
	else:
		opt = raw_input('These files will be compiled and linked!\nDo you really? [Y/n]: ')
		if opt == '' or opt == 'y' or opt == 'Y':
			find_recursively_matched_files( \
					mydir, suffix_list, function, True)
			print 'Done!'
		else:
			print 'Canceled!'
