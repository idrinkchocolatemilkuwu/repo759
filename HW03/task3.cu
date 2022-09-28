#include "vscale.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);

	//set up random number generators 
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min_a = -10.0, max_a = 10.0;
	const float min_b = 0.0, max_b = 1.0;
	std::uniform_real_distribution<float> dist_a(min_a, max_a);
	std::uniform_real_distribution<float> dist_b(min_b, max_b);

	//initialize two arrays of length n with random numbers
	float *a, *b;
	cudaMallocManaged((void **)&a, sizeof(int) * n);
	cudaMallocManaged((void **)&b, sizeof(int) * n);
	for (int i = 0; i < n; i++){
		a[i] = dist_a(generator);
		b[i] = dist_b(generator);
	}

	//reference:olcf.ornl.gov/wp-content/uploads/2019/06/06_Managed_Memory.pdf
	//explicit prefetching
	int device;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(a, sizeof(float) * n, device);
	cudaMemPrefetchAsync(b, sizeof(float) * n, device);

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//set up the number of blocks and the number of threads/block
	int num_threads = 512;
	int num_blocks = (n + num_threads - 1) / num_threads;

	//launch kernel and time the events
	cudaEventRecord(start);
	vscale<<<num_blocks, num_threads>>>(a, b, n);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the elasped time, first element, and the last element of the array b
	std::printf("%f\n%f\n%f\n", ms, b[0], b[n-1]);

	cudaFree(a);
	cudaFree(b);
	return 0;
}
	


