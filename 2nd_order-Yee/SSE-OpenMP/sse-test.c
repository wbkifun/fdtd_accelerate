#include <stdlib.h>
#include <stdio.h>

typedef float v4sf __attribute__((vector_size(16)));

int main() {
	printf("L1 cache line size = %d\n", CLS);
	int i, nx=100;
	float *a, *b, *c1, *c2;
	
	a = (float *) malloc(nx*sizeof(float));
	b = (float *) malloc(nx*sizeof(float));
	c1 = (float *) malloc(nx*sizeof(float));
	c2 = (float *) malloc(nx*sizeof(float));

	for(i=0; i<nx; i++) {
	   a[i] = i;	
	   b[i] = 2*i;	
	}

	// without sse
	for(i=0; i<nx; i++) c1[i] = a[i]+b[i];	

	// using sse
	v4sf va, vb, vc;
	for(i=0; i<nx; i+=4) {
		va = __builtin_ia32_loadups( &a[i] );
		vb = __builtin_ia32_loadups( &b[i] );
		vc = __builtin_ia32_addps( va, vb );
		__builtin_ia32_storeups(&c2[i], vc);
	}

	// result
	float errsum;
	for(i=0; i<nx; i++) errsum += c1[i] - c2[i];
	printf("errsum=%f\n", errsum);

	return 0;
}
