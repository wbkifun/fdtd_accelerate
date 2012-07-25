#!/usr/bin/env python

import subprocess as sp


npml_list = [8, 10, 16, 32, 64]
proc_list = []
for npml in npml_list:
	cmd = './110-npml_+x-min_r.py %d' % npml
	proc = sp.Popen(cmd.split())
	proc_list.append(proc)

for proc in proc_list:
	proc.wait()