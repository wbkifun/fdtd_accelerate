#!/usr/bin/env python
# _*_ coding: utf-8 _*_

"""
 <File Description>

 File Name : structure.py

 Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
          Kim, KyoungHo (rain_woo@korea.ac.kr)

 Written date : 2008. 3. 6

 Copyright : GNU GPL

============================== < File Description > ===========================

Define the functions for apply structure to matter-arrays.
    apply_structure
    write_structure_files
    read_structure_files

===============================================================================
"""

from structure_base import *
import glob
from scipy.io.numpyio import fwrite, fread
from set_mp_line_arrays import *
from set_mp_area_arrays import *

def apply_structure(space, structures, apply_grid_opt):
    #--------------------------------------------------------------------------
    # read/write the structure files
    #--------------------------------------------------------------------------
    number_structure_files = 0
    for s_number, structure in enumerate(structures):
        filename1 = './structures/%.3d_%s_info.py' \
                % (s_number, structure.__name__)
        filename2 = './structures/%.3d_%s_ispts_[xyz].scbin' \
                % (s_number, structure.__name__)
        flist = glob.glob(filename1) + glob.glob(filename2)
        number_structure_files += len(flist)

    if len(structures) != 0 and number_structure_files == 4*len(structures):
        print 'The structure files are founded.'
        use_switch = raw_input('Do you use this files?(Y/n) ')

        if use_switch == 'Y' or use_switch == 'y' or use_switch == '':
            pass
        else:
            use_switch = raw_input('Do you really recreate the structure \
                    files?(Y/n) ')
            if use_switch == 'Y' or use_switch == 'y' or use_switch == '':
                write_structure_files(structures)

    else:
        print 'There are no proper structure files.'
        write_structure_files(structures)

    structures = read_structure_files()


    if apply_grid_opt == 'line':
        set_mp_line_arrays(structures, space.matter_parameter_line_arrays)

    elif apply_grid_opt == 'area':
        set_mp_area_arrays(structures, space.matter_parameter_area_arrays)

    elif apply_grid_opt == 'both':
        set_mp_line_arrays(structures, space.matter_parameter_line_arrays)
        set_mp_area_arrays(structures, space.matter_parameter_area_arrays)


def write_structure_files(structures):
    print 'Writing the structure files...'
    for s_number, structure in enumerate(structures):
        print '%d) %s' %(s_number, structure.__name__)

        info_str = '#!/usr/bin/env python\n \
                __name__ = %s\n \
                matter = %s\n \
                DER = discretized_effective_region = %s\n \
                DERS = discretized_effective_sides = %s\n \
                DERC = discretized_effective_center = %s'
                % ( structure.__name__, \
                repr(structure.matter), \
                repr(structure.discretized_effective_region), \
                repr(structure.discretized_effective_sides), \
                repr(structure.discretized_effective_center) )
        filename = './structures/%.3d_%s_info.py' \
                % (s_number, structure.__name__)
        print filename
        info_file = open(filename, 'w')
        info_file.write(info_str)
        info_file.close()

        print 'Calculate the arrays of the intersection points...'
        structure.make_intersection_points_arrays()
        ISPTs_array_x = structure.intersection_points_arrays[0]
        ISPTs_array_y = structure.intersection_points_arrays[1]
        ISPTs_array_z = structure.intersection_points_arrays[2]

        filename = './structures/%.3d_%s_ispts_x.scbin' \
                % (s_number, structure.__name__)
        print filename
        array_file = open(filename, 'wb')
        fwrite(array_file, ISPTs_array_x.size, ISPTs_array_x)
        array_file.close()
        filename = './structures/%.3d_%s_ispts_y.scbin' \
                % (s_number, structure.__name__)
        print filename
        array_file = open(filename, 'wb')
        fwrite(array_file, ISPTs_array_y.size, ISPTs_array_y)
        array_file.close()
        filename = './structures/%.3d_%s_ispts_z.scbin' \
                % (s_number, structure.__name__)
        print filename
        array_file = open(filename, 'wb')
        fwrite(array_file, ISPTs_array_z.size, ISPTs_array_z)
        array_file.close()


def read_structure_files():
    print 'Reading the structure files...' 
    structures = []
    s_number = 0

    flist = glob.glob('./structures/*')
    for file in flist:
        if '_info' in file:
            print file
            structures.append( __import__(file) )
            structure = structures[-1]
            Nx, Ny, Nz = structure.discrete_effective_region_sides

            filename = './structures/%.3d_%s_ispts_x.scbin' \
                    % (s_number, structure.__name__)
            print filename
            array_file = open(filename, 'rb')
            ISPTs_array_x = fread(filename, 2*Ny*Nz, 'f').reshape(2,Ny,Nz)
            array_file.close()
            filename = './structures/%.3d_%s_ispts_y.scbin' \
                    % (s_number, structure.__name__)
            print filename
            array_file = open(filename, 'rb')
            ISPTs_array_y = fread(filename, 2*Nx*Nz, 'f').reshape(2,Nx,Nz)
            array_file.close()
            filename = './structures/%.3d_%s_ispts_z.scbin' \
                    % (s_number, structure.__name__)
            print filename
            array_file = open(filename, 'rb')
            ISPTs_array_z = fread(filename, 2*Nx*Ny, 'f').reshape(2,Nx,Ny)
            array_file.close()

            structure.intersection_points_arrays = [ \
                    ISPTs_array_x, ISPTs_array_y, ISPTs_array_z]
            s_number += 1

    return structures


