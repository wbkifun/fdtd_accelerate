#!/usr/bin/env python

update_e_template="""
	__SHx;
	s[tx+TPB] = hx[idx+(4*I+1)*TPB];
	s[tx+2*TPB] = hx[idx+(4*I+2)*TPB];
	s[tx+3*TPB] = hx[idx+(4*I+3)*TPB];
	s[tx+4*TPB] = hx[idx+(4*I+4)*TPB];
	__syncthreads();
	tmp_ezx[0] = s[tx+TPB] - s[tx];
	tmp_ezx[1] = s[tx+2*TPB] - s[tx+TPB];
	tmp_ezx[2] = s[tx+3*TPB] - s[tx+2*TPB];
	tmp_ezx[3] = s[tx+4*TPB] - s[tx+3*TPB];
	tmp_ey[0] = s[tx+1] - s[tx];
	tmp_ey[1] = s[tx+TPB+1] - s[tx+TPB];
	tmp_ey[2] = s[tx+2*TPB+1] - s[tx+2*TPB];
	tmp_ey[3] = s[tx+3*TPB+1] - s[tx+3*TPB];

	__SHy;
	s[tx+6*TPB] = hy[idx+(4*I+1)*TPB];
	s[tx+7*TPB] = hy[idx+(4*I+2)*TPB];
	s[tx+8*TPB] = hy[idx+(4*I+3)*TPB];
	s[tx+9*TPB] = hy[idx+(4*I+4)*TPB];
	__syncthreads();
	if( i<nx-1 && j<ny-1 ) {
		tmp_ce[0] = cez[idx+(4*I)*TPB];
		tmp_ce[1] = cez[idx+(4*I+1)*TPB];
		tmp_ce[2] = cez[idx+(4*I+2)*TPB];
		tmp_ce[3] = cez[idx+(4*I+3)*TPB];
		tmp_h[0] = hy[idx+(4*I)*TPB+nyz];
		tmp_h[1] = hy[idx+(4*I+1)*TPB+nyz];
		tmp_h[2] = hy[idx+(4*I+2)*TPB+nyz];
		tmp_h[3] = hy[idx+(4*I+3)*TPB+nyz];

		ez[idx+(4*I)*TPB] += tmp_ce[0]*( tmp_h[0] - s[tx+5*TPB] - tmp_ezx[0] );
		ez[idx+(4*I+1)*TPB] += tmp_ce[1]*( tmp_h[1] - s[tx+6*TPB] - tmp_ezx[1] );
		ez[idx+(4*I+2)*TPB] += tmp_ce[2]*( tmp_h[2] - s[tx+7*TPB] - tmp_ezx[2] );
		ez[idx+(4*I+3)*TPB] += tmp_ce[3]*( tmp_h[3] - s[tx+8*TPB] - tmp_ezx[3] );
	}
	tmp_ezx[0] = s[tx+5*TPB+1] - s[tx+5*TPB];
	tmp_ezx[1] = s[tx+6*TPB+1] - s[tx+6*TPB];
	tmp_ezx[2] = s[tx+7*TPB+1] - s[tx+7*TPB];
	tmp_ezx[3] = s[tx+8*TPB+1] - s[tx+8*TPB];

	__SHz;
	s[tx+11*TPB] = hz[idx+(4*I+1)*TPB];
	s[tx+12*TPB] = hz[idx+(4*I+2)*TPB];
	s[tx+13*TPB] = hz[idx+(4*I+3)*TPB];
	s[tx+14*TPB] = hz[idx+(4*I+4)*TPB];
	__syncthreads();
	if( i<nx-1 && k<nz-1 ) {
		tmp_ce[0] = cey[idx+(4*I)*TPB];
		tmp_ce[1] = cey[idx+(4*I+1)*TPB];
		tmp_ce[2] = cey[idx+(4*I+2)*TPB];
		tmp_ce[3] = cey[idx+(4*I+3)*TPB];
		tmp_h[0] = hz[idx+(4*I)*TPB+nyz];
		tmp_h[1] = hz[idx+(4*I+1)*TPB+nyz];
		tmp_h[2] = hz[idx+(4*I+2)*TPB+nyz];
		tmp_h[3] = hz[idx+(4*I+3)*TPB+nyz];

		ey[idx+(4*I)*TPB] += tmp_ce[0]*( tmp_ey[0] - tmp_h[0] + s[tx+10*TPB] );
		ey[idx+(4*I+1)*TPB] += tmp_ce[1]*( tmp_ey[1] - tmp_h[1] + s[tx+11*TPB] );
		ey[idx+(4*I+2)*TPB] += tmp_ce[2]*( tmp_ey[2] - tmp_h[2] + s[tx+12*TPB] );
		ey[idx+(4*I+3)*TPB] += tmp_ce[3]*( tmp_ey[3] - tmp_h[3] + s[tx+13*TPB] );
	}
	if( j<ny-1 && k<nz-1 ) {
		tmp_ce[0] = cex[idx+(4*I)*TPB];
		tmp_ce[1] = cex[idx+(4*I+1)*TPB];
		tmp_ce[2] = cex[idx+(4*I+2)*TPB];
		tmp_ce[3] = cex[idx+(4*I+3)*TPB];
		
		ex[idx+(4*I)*TPB] += tmp_ce[0]*( s[tx+11*TPB] - s[tx+10*TPB] - tmp_ezx[0] );
		ex[idx+(4*I+1)*TPB] += tmp_ce[1]*( s[tx+12*TPB] - s[tx+11*TPB] - tmp_ezx[1] );
		ex[idx+(4*I+2)*TPB] += tmp_ce[2]*( s[tx+13*TPB] - s[tx+12*TPB] - tmp_ezx[2] );
		ex[idx+(4*I+3)*TPB] += tmp_ce[3]*( s[tx+14*TPB] - s[tx+13*TPB] - tmp_ezx[3] );
	}

"""
tpb = 10
ny, nz = 2, 3

o1 = ['__SHx', '__SHy', '__SHz']
r1 = ['s[tx] = hx[idx]', 's[tx+5*TPB] = hy[idx]', 's[tx+10*TPB] = hz[idx]']
r2 = ['s[tx] = s[tx+4*TPB]', 's[tx+5*TPB] = s[tx+9*TPB]', 's[tx+10*TPB] = s[tx+14*TPB]']

template = update_e_template
for i in range(1):
	if(i == 0): 
		for o, r in zip(o1, r1): template = template.replace(o,r)
	else:
		for o, r in zip(o1, r2): template = template.replace(o,r)

	template = template.replace('(4*I)*TPB+nyz', str( (4*i)*tpb+ny*nz ))
	template = template.replace('(4*I)*TPB', str( (4*i)*tpb ))
	for j in range(1,5):
		template = template.replace('(4*I+%s)*TPB+nyz' % str(j), str( (4*i+j)*tpb+ny*nz ))
		template = template.replace('(4*I+%s)*TPB' % str(j), str( (4*i+j)*tpb ))
	for j in range(2,14):
		template = template.replace('%s*TPB' % str(j), str( j*tpb ))

	template = template.replace('TPB', str(tpb))

print template
