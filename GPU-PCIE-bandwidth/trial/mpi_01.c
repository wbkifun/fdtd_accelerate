#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MASTER		0
#define CHUNKSIZE	10

MPI_Status status;

int main( int argc, char **argv ) {
	int rank, size, Nnode;
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &size );
	Nnode = size - 1;

	int i;
	float *data;

	if ( rank == MASTER ) {
		printf("MASTER: number of worker tasks will be= %d\n", Nnode);

		int index, dest;
		int Nx = CHUNKSIZE*Nnode;
		float *result;
		data = (float *)calloc( Nx, sizeof(float) ); 
		result = (float *)calloc( Nx, sizeof(float) ); 

		index = 0;
		for ( dest=1; dest<= Nnode; dest++ ) {
			printf("Sending to worker task= %d\n", dest);
			MPI_Send( &data[index], CHUNKSIZE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD );
			index += CHUNKSIZE;
		}

		index = 0;
		for ( dest=1; dest<= Nnode; dest++ ) {
			printf("Receive from worker task= %d\n", dest);
			MPI_Recv( &result[index], CHUNKSIZE, MPI_FLOAT, dest, 1, MPI_COMM_WORLD, &status );
			index += CHUNKSIZE;
		}

		printf("Result:\n");
		for ( i=0; i<Nx; i++ ) printf("%g ", result[i] ); 
		printf("\n");
		printf("MASTER: All Done!\n");
	}

	else {
		data = (float *)calloc( CHUNKSIZE, sizeof(float) ); 
		//printf("Receive from master task= %d\n", rank);
		MPI_Recv( data, CHUNKSIZE, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status );
		for ( i=0; i<CHUNKSIZE; i++ ) data[i] += rank;

		//printf("Send to master task= %d\n", rank);
		MPI_Send( data, CHUNKSIZE, MPI_FLOAT, MASTER, 1, MPI_COMM_WORLD );
	}

	MPI_Finalize();
	return 0;
}
