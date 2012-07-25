#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


void *print_hello( void *threadid ) {
	long tid;
	tid = (long)threadid;
	printf("Hello World! It's me, thread #%ld!\n", tid);
	pthread_exit(NULL);
}


void func( int Ncore ) {
	pthread_t threads[Ncore];
	int rc;
	long t;
	for ( t=0; t<Ncore; t++ ) {
		printf("In main: craeting thread %ld\n", t );
		rc = pthread_create( &threads[t], NULL, print_hello, (void *)t );
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n",rc);
			exit(0);
		}
	}
}


int main( int argc, char *argv[] ) {
	int Ncore = 5;
	func( Ncore );

	pthread_exit(NULL);
}
