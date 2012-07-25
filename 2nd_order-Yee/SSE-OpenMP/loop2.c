#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
	int i;
	int N = 10;
	float *a, *b, *c;
	a = calloc ( N, sizeof(float) );
	b = calloc ( N, sizeof(float) );
	c = calloc ( N, sizeof(float) );

	for ( i=0; i<N; i++ ) {
		a[i] = i;
		b[i] = 0.1*i;
	}

	#pragma omp parallel for
	for ( i=0; i<N; i++ ) {
		c[i] = a[i]*a[i] + b[i]*b[i];
	}

	for ( i=0; i<N; i++ ) printf("%g\t", c[i]);
	printf("\n");
}
