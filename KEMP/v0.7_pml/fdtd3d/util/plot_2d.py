import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from glob import glob
from optparse import OptionParser
from time import sleep


# parsing the arguments
usage = "usage: %prog [options] h5_dir_path"
parser = OptionParser(usage=usage)
parser.add_option('-t', '--tag', type='int', dest='tag', default=0, help='tag number')
parser.add_option('-f', '--field', action='store', type='string', dest='str_f', help='target field (ex, ey, ez, hx, hy, hz)')
parser.add_option('-s', '--sleep', type='int', dest='sleep_time', default=1, help='sleep time interval for while loop')
parser.add_option("--no_device_lines", action="store_false", dest="no_device_lines", default=False)
parser.add_option("--save_png", action="store_true", dest="save_png", default=False)
(opts, args) = parser.parse_args()

try:
    dir_path = args[0] if args[0].endswith('/') else args[0]+'/'
except IndexError:
    parser.error('h5_dir_path is missing.')

tag = opts.tag
str_f = opts.str_f
sleep_time = opts.sleep_time
off_device_lines = opts.no_device_lines
on_save_png = opts.save_png


# parsing info_dict
exist_fpath = True
while exist_fpath:
    try:
        pkl_file = open(dir_path + 'divide_info.pkl', 'rb')
        info_dict = pickle.load(pkl_file)
        pkl_file.close()
        exist_fpath = False
    except:
        sleep(sleep_time)

shape = info_dict['shape']
assert len(shape) == 2, 'The target array must be two-dimension. %d is given.' % len(shape)

strf = info_dict['str_fs'][0] if str_f == None else str_f
pt0 = info_dict['pt0']
pt1 = info_dict['pt1']
tmax = info_dict['tmax']
tgap = info_dict['tgap']
anx_list = info_dict['anx_list']

is_mpi = True if info_dict.has_key('ranks') else False
if is_mpi:
    rank_list = info_dict['ranks']
    slices_list = [info_dict[rank] for rank in rank_list]
    any_list = info_dict['any_list']
    anz_list = info_dict['anz_list']
    ans_dict = {'x': anx_list, 'y': any_list, 'z': anz_list}
else:
    ans_dict = {'x': anx_list}

nx, ny = shape
axes = ['x', 'y', 'z']
shape_dict = dict( zip(axes, shape) )
axis0, axis1 = ''.join( [ax for ax, p0, p1 in zip(axes, pt0, pt1) if p0 != p1] )


# prepare the plot
plt.ion()
fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(1, 1, 1)

if not off_device_lines:
    for i in ans_dict.get(axis0, [])[1:-1]:
        ax1.plot((i, i), (0, shape_dict[axis1]), color='k', linewidth=1.2)

    for i in ans_dict.get(axis1, [])[1:-1]:
        ax1.plot((0, shape_dict[axis0]), (i, i), color='k', linewidth=1.2)

imag = ax1.imshow(np.ones(shape, np.float32).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
ax1.set_xlabel(axis0)
ax1.set_ylabel(axis1)
plt.colorbar(imag)


# plot
ndigit = int( ('%e' % tmax).split('+')[1] ) + 1
fpath_form = dir_path + '%%.%dd_tag%d.h5' % (ndigit, tag)
if is_mpi:
    fpath_form_list = [fpath_form.rstrip('.h5') + '_mpi%d.h5' % rank for rank in rank_list]
    tmp_arr = np.zeros(shape, np.float32)
else:
    fpath_form_list = [fpath_form]

for tstep in xrange(tgap, tmax+1, tgap):
    fpath_list = [fpath_form % tstep for fpath_form in fpath_form_list]
    h5f_list = []
    for fpath in fpath_list:
        exist_fpath = True
        while exist_fpath:
            try:
                h5f_list.append( h5.File(fpath, 'r') )
                exist_fpath = False
            except:
                sleep(sleep_time)

    if is_mpi:
        for slices, h5f in zip(slices_list, h5f_list):
            tmp_arr[slices] = h5f[strf].value
        imag.set_array(tmp_arr.T)
    else:
        imag.set_array(h5f_list[0][strf].value.T)

    plt.title(strf + ' tstep= %s' % tstep)
    if on_save_png:
        png_path_form = dir_path + 'png/%%.%dd.png'
        plt.savefig(png_path_form % tstep_str)
    plt.draw()

sleep(10)
