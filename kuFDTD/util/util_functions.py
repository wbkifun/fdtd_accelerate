#!/usr/bin/env python

import os

def find_recursively_matched_files(dir, suffix_list, func, execute_opt):
	filelist = os.listdir(dir)
	matched_file_count = 0

	for file in filelist:
		fpath = dir + '/' + file
		if os.path.isfile(fpath):
			if file[file.rfind('.'):] in suffix_list:
				matched_file_count += 1
				if execute_opt:
					#print fpath
					func(dir, file)
				else:
					print fpath
		elif os.path.isdir(fpath):
			matched_file_count += \
					find_recursively_matched_files( \
					fpath, suffix_list, func, execute_opt) 
	
	return matched_file_count
