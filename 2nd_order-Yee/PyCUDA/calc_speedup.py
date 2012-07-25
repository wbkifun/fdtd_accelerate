#!/usr/bin/env python

import sys

t1 = float(sys.argv[1])
t2 = float(sys.argv[2])

reduced_time_percentage = (t1-t2)/t1
speedup = t1/t2

print "reduced time: %2.3f times" % reduced_time_percentage
print 'speedup:      %2.3f times' % speedup
