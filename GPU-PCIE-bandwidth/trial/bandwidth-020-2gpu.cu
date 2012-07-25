#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


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
		for ( j=0; j<TMAX; j++ ) 
			cudaMemcpy ( devA, A, size, cudaMemcpyHostToDevice );
		dt = elapsed_time( t1, time_str );
		//printf("%s\n", time_str);
		drate += size*TMAX/gbyte/dt;
	}

	return drate/10;
}


int main() {
	long NG = gbyte/sizeof(float);
	long NM = mbyte/sizeof(float);

	long Nx = 1*NG;
	long size = Nx*sizeof(float);
	float **A;
	float **devA;

	int i;

	int num_gpus=1, gpu_id;
	//cudaGetDeviceCount( &num_gpus );
	printf("The Number of GPU devices: %d\n", num_gpus );

	A = (float **) malloc ( num_gpus*sizeof(float *) );
	devA = (float **) malloc ( num_gpus*sizeof(float *) );
	for ( gpu_id=0; gpu_id<num_gpus; gpu_id++ ) 
		A[gpu_id] = (float *) malloc( size );

	int err;
	for ( gpu_id=0; gpu_id<num_gpus; gpu_id++ ) {
		cudaSetDevice( gpu_id );
		err = cudaMalloc ( (void **) &devA[gpu_id], size );
		if(err) {
			printf("Error (%d): cudaMalloc is failed in dev %d\n", err, gpu_id);
			exit(0);
		}
	}

	for ( gpu_id=0; gpu_id<num_gpus; gpu_id++ ) for ( i=0; i<Nx; i++ ) A[gpu_id][i] = rand();

	omp_set_num_threads( num_gpus );
	#pragma omp parallel
	{
	   	unsigned int cpu_thread_id = omp_get_thread_num(); 
		unsigned int num_cpu_threads = omp_get_num_threads();

		int gpu_id = -1;
	   	cudaSetDevice( cpu_thread_id % num_gpus );
	   	cudaGetDevice( &gpu_id ); 

		printf("[GPU %d]Data rate: %1.2f GB/s\n", gpu_id, measure_data_rate( devA[gpu_id], A[gpu_id], size ) );
	}

	cudaThreadExit();

	return 0;
}
