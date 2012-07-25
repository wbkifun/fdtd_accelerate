#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>


void elapsed_time( struct timeval t1, char str[] ) {
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
}


int main() {
	struct timeval t1;
	char time_str[32];
	int i, j, a;

	gettimeofday( &t1, NULL );

	for( i=0; i<100000; i++) {
		for( j=0; j<10000; j++) {
			a = j;
		}
	}

	elapsed_time( t1, time_str );
	printf("%s\n", time_str);

	return 0;
}
