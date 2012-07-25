#!/usr/bin/env python

import os
import sys

mydir = '/home/kifang/kuFDTD/'
remote_dir = 'kufdtd@nolftp:/nolftp/kuFDTD/'

try: 
    operation = sys.argv[1]
except IndexError:
    operation = None

if operation == 'get':
    cmd = 'rsync -arvu -e ssh %s %s' %(remote_dir, mydir)
    print cmd
    os.system(cmd)

elif operation == 'put':
    cmd = 'rsync -arvu -e ssh %s %s' %(mydir, remote_dir)
    print cmd
    os.system(cmd)

else:
    print 'Usage: ./rsync_kuFDTD.py [get/put]'
