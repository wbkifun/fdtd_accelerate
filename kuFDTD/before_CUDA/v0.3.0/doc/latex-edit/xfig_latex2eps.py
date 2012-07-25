#!/usr/bin/env python

import sys
import os

input_file = sys.argv[1]
if not input_file.endswith('pstex_t'):
    print 'Error: Wrong pstex_t file.'
    sys.exit()

f = file('xfig_latex2eps.tex.org','r')
s = f.read()
s = s.replace('pstex_t_file',input_file)
f.close()

f = file('xfig_latex2eps.tex','w')
f.write(s)
f.close()


output_file = '%s.eps' % input_file.rstrip('.pstex_t')
print output_file

cmd = 'latex xfig_latex2eps.tex && dvips -E xfig_latex2eps.dvi -o %s' % output_file
os.system(cmd)
