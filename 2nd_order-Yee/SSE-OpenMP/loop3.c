#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
	int i,j,k;
	int N = 10000;
	float *a, *b, *c;
	a = calloc ( N, sizeof(float) );
	b = calloc ( N, sizeof(float) );
	c = calloc ( N, sizeof(float) );

	for ( i=0; i<N; i++ ) {
		a[i] = i;
		b[i] = 0.1*i;
	}

	omp_set_num_threads(2);
	#pragma omp parallel for private(i)
	for ( i=0; i<N; i++ ) {
		for ( j=0; j<N; j++ ) {
			for ( k=0; k<N; k++ ) {
			c[i] = a[i]*a[i] + b[i]*b[i];
			}
		}
	}

	for ( i=0; i<N; i++ ) printf("%g\t", c[i]);
	printf("\n");
}
