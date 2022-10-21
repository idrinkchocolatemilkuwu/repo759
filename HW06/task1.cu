#include "mmul.h"
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <cublas_v2.h>

int main(int argc, char** argv) {

	int n = atol(argv[1]);
	int n_tests = atol(argv[2]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist_a(min, max);
	std::uniform_real_distribution<float> dist_b(min, max);
	std::uniform_real_distribution<float> dist_c(min, max);

	//create A, B and C
	//fill with random numbers
	float* A, * B, * C;
	cudaMallocManaged((void**)&A, n * n * sizeof(float));
	cudaMallocManaged((void**)&B, n * n * sizeof(float));
	cudaMallocManaged((void**)&C, n * n * sizeof(float));
	for (int i = 0; i < n * n; i++) {
		A[i] = dist_a(generator);
		B[i] = dist_b(generator);
		C[i] = dist_c(generator);
	}

	//set up cublas
	cublasHandle_t handle;
	cublasCreate(&handle);

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//call mmul n_tests times and time it
	cudaEventRecord(start);
	for (int i = 0; i < n_tests; i++) {
		mmul(handle, A, B, C, n);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the average time taken by a single call to mmul
	float average_ms = ms / n_tests;
	printf("%f", average_ms);

	cudaFree(A); cudaFree(B); cudaFree(C);
	cublasDestroy(handle);
	return 0;
}