#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : renames.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)

 Written date : 2008. 10. 24

 Copyright : GNU GPL2

============================== < File Description > ===========================

rename the files or the directories matched given string at current directory

===============================================================================
"""

import sys
import os
import glob

try:
	pre_str = sys.argv[1]
	post_str = sys.argv[2]
except IndexError:
	print 'Error: Wrong arguments'
	print 'Usage: ./removes.py pre_string post_string'
	sys.exit()
	 
path_list = glob.glob('./*')

count_file = 0
count_dir = 0
for path in path_list:
	if pre_str in path:
		post_path = path.replace(pre_str, post_str)
		print path
		if os.path.isfile(path):
			count_file += 1
		elif os.path.isdir(path):
			count_dir += 1

print '-'*80
print 'There are %d files and %d directories to rename.' % (count_file, count_dir)
which_rename = raw_input('Do you rename? (Y/n): ')

if which_rename == '' or which_rename == 'Y' or which_rename == 'y':
	for path in path_list:
		if pre_str in path:
			post_path = path.replace(pre_str, post_str)
			print 'rename %s %s' %(path, post_path)
			os.rename(path, post_path)
	print 'Done.'
else:
	sys.exit()
