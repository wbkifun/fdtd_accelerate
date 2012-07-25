#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>


long gbyte = 1024*1024*1024;
long mbyte = 1024*1024;


float elapsed_time( struct timeval t1, char str[] ) {
	struct timeval t2;
	long dt, dut;
	int dd, dh, dm, ds;

	gettimeofday( &t2, NULL );
	dt = t2.tv_sec - t1.tv_sec;
	dut = t2.tv_usec - t1.tv_usec;
	if ( dut < 0 ) {
		dt -= 1;
		dut += 1e6;
	}
	dd = dt/86400;
	dh = dt%86400/3600;
	dm = dt%3600/60;
	ds = dt%60;

	sprintf( str, "[%.2d]%.2d:%.2d:%.2d.%.6ld", dd, dh, dm, ds, dut );

	return ( dt + dut*1e-6 );
}

float measure_data_rate( float *devA, float *A, long size ) {
	struct timeval t1;
	char time_str[32];
	float dt;
	int TMAX = 100;
	int i, j;

	float drate=0;
	for ( i=0; i<10; i++ ) {
		gettimeofday( &t1, NULL );
		for ( j=0; j<TMAX; j++ ) cudaMemcpy ( devA, A, size, cudaMemcpyHostToDevice );
		dt = elapsed_time( t1, time_str );
		printf("%s\n", time_str);
		drate += size*TMAX/gbyte/dt;
	}

	return drate/10;
}


int main() {
	struct timeval t1;
	char time_str[32];
	int i;

	long NG = gbyte/sizeof(float);
	long NM = mbyte/sizeof(float);

	long Nx = 500*NM;
	long size = Nx*sizeof(float);
	float *A;
	float *devA;
	A = (float *) malloc( size );

	int err;
	err = cudaMalloc ( (void **) &devA, size );
	if(err) {
		printf("Error (%d): cudaMalloc is failed\n", err);
		exit(0);
	}

	gettimeofday( &t1, NULL );
	for ( i=0; i<Nx; i++ ) A[i] = rand();
	elapsed_time( t1, time_str );
	printf("%s\n", time_str);

	printf("Data rate: %1.2f GB/s\n", measure_data_rate( devA, A, size ) );

	return 0;
}
