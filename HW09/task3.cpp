#include "mpi.h"
#include <iostream>
#include <chrono>
#include <random>
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
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	duration<double, milli> t0, t1;

	//implement the procedure listed
	if (rank == 0){
		//start timing t0
		chrono::high_resolution_clock::time_point start;
		chrono::high_resolution_clock::time_point end;
		start = chrono::high_resolution_clock::now();
		//rank 0 should send array "A" and receive array "B"
		MPI_Send(A, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(B, n, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
		end = chrono::high_resolution_clock::now();
		//measure t0
		t0 = std::chrono::duration_cast<duration<double, milli>>(end - start);
		//receive t1
		MPI_Recv(&t1, 1, MPI_DOUBLE, 1,  2, MPI_COMM_WORLD, &status);
		//print t0 + t1
		printf("%f\n", t0 + t1);

	}else if (rank == 1){
		//start timing t1
		chrono::high_resolution_clock::time_point start;
		chrono::high_resolution_clock::time_point end;
		start = chrono::high_resolution_clock::now();
		//rank 1 should receive array "A" and receive array "B"
		MPI_Recv(A, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Send(B, n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
		end = chrono::high_resolution_clock::now();
		//measure t1
		t1 = std::chrono::duration_cast<duration<double, milli>>(end-start);
		//send t1
		MPI_Send(&t1, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}

