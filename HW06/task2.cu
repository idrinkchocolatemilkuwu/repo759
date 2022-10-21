#include "scan.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

int main(int argc, char** argv) {

	int n = atol(argv[1]);
	int threads_per_block = atol(argv[2]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist(min, max);

	//create input output
	//fill with random numbers
	float* input, *output;
	cudaMallocManaged((void**)&input, n * sizeof(float));
	cudaMallocManaged((void**)&output, n * sizeof(float));
	for (int i = 0; i < n; i++) {
		input[i] = dist(generator);
	}

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//call the kernel and time it
	cudaEventRecord(start);
	scan(input, output, n, threads_per_block);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the last element and the time
	printf("%f\n%f", output[n - 1], ms);

	cudaFree(input); cudaFree(output);
	return 0;
}