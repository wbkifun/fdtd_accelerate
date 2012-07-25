#!/usr/bin/env python

import numpy as np

f = open('prediction_experiment_results.dat')
head = f.readline()
values = []
for line in f:
	values.append(line.rstrip('\n').split('\t'))
f.close()

arr = np.float32(values)
errors = np.fabs(arr[:,0]-arr[:,1])/arr[:,0]*100
print errors
print errors.mean()
