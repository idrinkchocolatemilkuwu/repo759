#include <mpi.h>
#include <iostream>
#include <chrono>
#include <random>
#include <stdio.h>
using namespace std;
using chrono::duration;

//reference: www.umsl.edu/~siegelj/CS4740_5740/Algorithmsll/MPI_send_receive.html
//piazza post 445

int main(int argc, char** argv){

	int n = atol(argv[1]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist(min,max);

	//create and fill two buffer arrays
	float* A = (float*) malloc (n * sizeof(float));
	float* B = (float*) malloc (n * sizeof(float));
	for (int i = 0; i < n; i++){
		A[i] = dist(generator);
		B[i] = dist(generator);
	}

	//initialize variables necessary for MPI_Send and MPI_Recv
	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	//duration<double, milli> t0, t1;
	double t0, t1;

	//implement the procedure listed
	if (rank == 0){
		//start timing t0
		double start0, end0;
		start0 = MPI_Wtime();
		//rank 0 should send array "A" and receive array "B"
		MPI_Send(A, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(B, n, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
		end0 = MPI_Wtime();
		//measure t0
		t0 = (end0 - start0) * 1000;
		//send t0
		MPI_Send(&t0, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
		//receive t1
		//MPI_Recv(&t1, 1, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD, &status);

	}else if (rank == 1){
		//start timing t1
		double start1, end1;
		start1 = MPI_Wtime();
		//rank 1 should receive array "A" and send array "B"
		MPI_Recv(A, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Send(B, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
		end1 = MPI_Wtime();
		//measure t1
		t1 = (end1 - start1) * 1000;
		//receive t0
		MPI_Recv(&t0, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
		//send t1
		//MPI_Send(&t1, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

		//print t0 + t1
		printf("%f\n", t0 + t1);
		//std::cout << t0 + t1 << std::endl;
	}
	//print t0 + t1
	//printf("%f\n", t0 + t1);

	MPI_Finalize();
}

