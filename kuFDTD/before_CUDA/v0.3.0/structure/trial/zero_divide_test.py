#!/usr/bin/env python

import scipy as sc

for d in sc.arange(0, 1.01, 0.01):
	for ep1 in sc.arange(1, 10.01, 0.01):
		for ep2 in sc.arange(1, 10.01, 0.01):
			B = d/ep1 + (1-d)/ep2
			A = (B**2 + 1)/(B**2 - 1)

			if A >= -1 and A <= 1:
				print 'A=%g, (B=%g, d=%g, ep1=%g, ep2=%g)' % (A, B, d, ep1, ep2)
