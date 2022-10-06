#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

int main(int argc, char **argv){
	size_t n = atol(argv[1]);
	size_t threads_per_block = atol(argv[2]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist_A(min,max);
	std::uniform_real_distribution<float> dist_B(min,max);

	//initialize arrays of length n * n A and B with random numbers
	float *A, *B, *C;
	cudaMallocManaged((void **)&A, sizeof(float) * n * n);
	cudaMallocManaged((void **)&B, sizeof(float) * n * n);
	cudaMallocManaged((void **)&C, sizeof(float) * n * n);
	for (size_t i = 0; i < n * n; i++){
		A[i] = dist_A(generator);
		//printf("%f ", A[i]);
		B[i] = dist_B(generator);
		//printf("%f ", B[i]);
	}

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//launch kernel and time the events
	cudaEventRecord(start);
	matmul(A, B, C, n, threads_per_block);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the last element of C and elapsed time
	printf("%f\n%f", C[n * n - 1], ms);

	//for (size_t i = 0; i < n * n; i++){
	//	printf("%f ", C[i]);
	//}

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	return 0;
}


