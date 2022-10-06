#include "stencil.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

int main(int argc, char **argv){
	unsigned int n = atol(argv[1]);
	unsigned int R = atol(argv[2]);
	unsigned int threads_per_block = atol(argv[3]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist_image(min,max);
	std::uniform_real_distribution<float> dist_mask(min,max);

	//initialize the arrays
	float *image, *output, *mask;	
	cudaMallocManaged((void **)&image, n * sizeof(float));
	cudaMallocManaged((void **)&output, n * sizeof(float));
	cudaMallocManaged((void **)&mask, (2 * R + 1) * sizeof(float));
	for (unsigned int i = 0; i < n; i++){
		image[i] = dist_image(generator);
	}
	for (unsigned int i = 0; i < 2 * R + 1; i++){
		mask[i] = dist_mask(generator);
	}

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//launch kernel and time the events
	cudaEventRecord(start);
	stencil(image, mask, output, n, R, threads_per_block);
	cudaEventRecord(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the last element of the output array and elapsed time
	printf("%f\n%f\n", output[n-1], ms);

	cudaFree(image);
	cudaFree(output);
	cudaFree(mask);
	return 0;
}
