#!/usr/bin/env python
#! _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : kufdtd_simulator.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 03. 25. Thu

 Copyright : GNU GPL

============================== < File Description > ===========================

The kufdtd main simulator except for the MPI parallel library.

===============================================================================
"""

from kufdtd_base import *
import sys
import os

#------------------------------------------------------------------------------
# import the project file
#------------------------------------------------------------------------------
try:
    proj_filename = sys.argv[1]
except IndexError:
    print 'Error: No project config file.'
    print 'Usage: kufdtd_simulator my_project.proj.py'
    sys.exit()
if !proj_filename.endswith('.proj.py'):
    print 'Error: No project config file.'
    print 'Usage: kufdtd_simulator my_project.proj.py'
else:
    di = proj_filename.rfind('/') # directory '/' index
    proj_dir = proj_filename[:di+1]
    proj_name = proj_filename[di+1:-8] 
    os.cddir(proj_dir)
    proj = __import__(proj_name+'.proj.py')

#------------------------------------------------------------------------------
# import the project file
#------------------------------------------------------------------------------
dimension = proj.dimension
if dimension 


