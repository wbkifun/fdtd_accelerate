from kufdtd.common import *
from pylab import *


class ViewGraphic2d:
	def __init__( s, data_shape, args, extra_args=[] ):
		ion()
		figure()

		s.imsh = imshow( transpose( sc.ones( data_shape, 'f' ) ), 
				cmap=cm.jet, 
				vmin=-0.05, vmax=0.05, 
				origin='lower', 
				interpolation='bilinear' )

		if 'colorbar' in extra_args:
			colorbar()


	def draw( s, data ):
		s.imsh.set_array( transpose(data) )
		draw()


	def save( s, path ):
		savefig( path )



class SaveData:
	pass
