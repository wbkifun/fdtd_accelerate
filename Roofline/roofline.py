#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import *

# ---------------------------------------------------------
# NVIDIA GPU Specifications
# ---------------------------------------------------------
# CC		(Compute Capability)
# SM		(Streaming Multiprocessor)
# SP		(Scalar Processor/SM)
# SFU		(Special Function Unit/SM)
# FPU/SFU	(Float Pointing Unit/SFU)
# F/C/P0	(flop/cycle/P0)		Port 0 instructions can be multiply-add (2 flop/cycle)
# F/C/P1	(flop/cycle/P1)		Port 1 instructions are just multiplies (1 flop/cycle)
# PCLK		(Processor Clock Speed)		[GHz]
# MCLK		(Memory Clock Speed)		[GHz]
# MW		(Memory Width)				[Bits/Channel]
# MC		(Memory Channel)
# ---------------------------------------------------------
gf9800 = {'name':'NVIDIA GeForce 9800 GTX+', 
		'CC':1.1, 'SM':16, 'SP':8, 'SFU':2, 'FPU/SFU':4, 
		'PCLK':1.836, 'MCLK':1.1, 'MW':256, 'MC':2}
c1060 = {'name':'NVIDIA TESLA C1060', 
		'CC':1.3, 'SM':30, 'SP':8, 'SFU':2, 'FPU/SFU':4, 
		'PCLK':1.300, 'MCLK':0.8, 'MW':512, 'MC':2}
gtx280 = {'name':'NVIDIA GeForce GTX 280', 
		'CC':1.3, 'SM':30, 'SP':8, 'SFU':2, 'FPU/SFU':4, 
		'PCLK':1.460, 'MCLK':1.107, 'MW':512, 'MC':2}
gtx480 = {'name':'NVIDIA GeForce GTX 480', 
		'CC':2.0, 'SM':15, 'SP':32, 'SFU':4, 'FPU/SFU':4, 
		'PCLK':1.401, 'MCLK':1.848, 'MW':384, 'MC':2}
# ---------------------------------------------------------


def calc_peak(dev, opt=[]):
	if 'noMAD' in opt:
		port0_throughput = dev['PCLK']*dev['SM']*dev['SP']
	else:
		port0_throughput = dev['PCLK']*dev['SM']*dev['SP']*2

	if 'noSFU' in opt:
		port1_throughput = 0
	else:
		port1_throughput = dev['PCLK']*dev['SM']*dev['SFU']*dev['FPU/SFU']

	pflops = port0_throughput + port1_throughput	# Peak FLOPS (GFLOPS)
	pmband = dev['MCLK']*dev['MW']*dev['MC']/8		# Peak Memory Bandwidth (GB/s)
	ipt = pflops/pmband								# intersection point

	return (pflops, pmband, ipt)


dev = c1060
print '<' + dev['name'] +'>'
pflops, pmband, ipt = calc_peak(dev)
print 'Peak Memory Bandwidth: %g GB/s' % pmband

print 'Peak FLOPS: %g GFLOPS' % pflops
print 'Intersection Point: %g FLOP/Byte' % ipt
pflops1, pmband, ipt1 = calc_peak(dev, ['noSFU'])
print 'Without SFU multiplies'
print '\tPeak FLOPS: %g GFLOPS' % pflops1
print '\tIntersection Point: %g FLOP/Byte' % ipt1
pflops2, pmband, ipt2 = calc_peak(dev, ['noSFU','noMAD'])
print 'Without SFU multiplies and MAD'
print '\tPeak FLOPS: %g GFLOPS' % pflops2
print '\tIntersection Point: %g FLOP/Byte' % ipt2

# ---------------------------------------------------------
# 3D FDTD Operational Intensity
# ---------------------------------------------------------
# ex = ex + cex*(hz - hz - hy + hy)
# <floating point operations>
# Add		: 2 * 3
# Subtract	: 2 * 3
# Multiply	: 1 * 3
# ---------------------
# Total		: 5 * 3 = 15	FLOP	-> 3 equations
#
# <DRAM traffic>
# Load		: 3 + 3 + 3
# Store		: 3
# ---------------------
# Total		: 12*(4 Byte) = 48	Byte
OI = 15/48.
ipt_y = pmband*OI
print '\nOperational Intensity: %g FLOP/Byte' % OI
print 'Attainable FLOPS: %g GFLOPS' % ipt_y
# ---------------------------------------------------------


xmin, xmax = 2**-4, 2**6
ymin, ymax = 2**2, 2**11

fig = figure(dpi=150)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(xmin,xmax), ylim=(ymin,ymax) )
ax.set_xscale("log", basex=2)
ax.set_yscale("log", basey=2)
ax.set_xticklabels(['1/16','1/8', '1/4', '1/2', '1', '2', '4', '8', '16', '32', '64'])
ax.set_yticklabels(['', '8', '16', '32', '64', '128', '256', '512', '1024', '2048'])
#ax.set_title('Roofline Model for %s' % dev['name'])
ax.set_xlabel('Operational Intensity (FLOP/Byte)')
ax.set_ylabel('Attainable FLOPS (GFLOPS)')
ax.grid()

ax.plot([xmin, ipt, xmax], [xmin*pmband, pflops, pflops], lw=2, color='gray')
ax.plot([xmin, ipt1, xmax], [xmin*pmband, pflops1, pflops1], lw=2, color='black')
ax.plot([ipt2, xmax], [pflops2, pflops2], lw=2, color='gray')

ax.text(0.1, 0.1*pmband*1.2, 'Peak Memory Bandwidth', color='gray', rotation=41)
ax.text(ipt*1.1, pflops*1.05, 'Peak FLOPS (Single)', color='gray')
ax.text(ipt1*1.5, pflops1*1.05, 'w/out SFU multiplies', color='gray')
ax.text(ipt2*1.5, pflops2*1.05, 'w/out MAD', color='gray')

ax.plot([OI,OI], [ymin,ipt_y], lw=3, ls='--', c='black')
ax.text(OI*0.8, ymin*1.2, '3D FDTD', color='gray', rotation=90)

'''
ax.plot([ipt,ipt], [pflops/4.5,pflops], lw=1.5, ls='--', c='black')
ax.text(ipt/1.15, pflops/4, '%1.2f'%ipt, rotation=90)
ax.plot([ipt/4.5,ipt], [pflops,pflops], lw=1.5, ls='--', c='black')
ax.text(ipt/4, pflops*1.05, '%d'%pflops)
'''

show()
