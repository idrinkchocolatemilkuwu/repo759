#include "count.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

int main(int argc, char** argv) {

	int n = atol(argv[1]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const int min = 0, max = 500;
	std::uniform_int_distribution<int> dist(min, max);

	//create host vector
	//fill with random numbers
	thrust::host_vector<int> h_in(n);
	for (int i = 0; i < n; i++) {
		h_in[i] = dist(generator);
	}

	//copy host vector into device vector
	//reference: https://stackoverflow.com/questions/43982841/how-to-copy-host-vector-to-device-vector-by-thrust
	thrust::device_vector<int> d_in = h_in;

	//create values and counts
	thrust::device_vector<int> values(n);
	thrust::device_vector<int> counts(n);

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//call count and time it
	cudaEventRecord(start);
	count(d_in, values, counts);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the last element of values and counts and time
	//printf("%i\n%i\n%f", values.back(), counts.back(), ms);
	std::cout << values.back() << std::endl;
	std::cout << counts.back() << std::endl;
	std::cout << ms << std::endl;

	return 0;
}