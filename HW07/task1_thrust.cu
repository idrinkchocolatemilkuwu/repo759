#include <cuda.h>
#include <stdio.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

//reference: https://docs.nvidia.com/cuda/thrust/index.html section 2, 3.2

int main(int argc, char** argv) {

	int n = atol(argv[1]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist(min, max);

	//create a thrust::host_vector of length n
	//fill with random numbers
	thrust::host_vector<float> H(n);
	for (int i = 0; i < n; i++) {
		H[i] = dist(generator);
	}

	//copy the thrust::host_vector into thrust::device_vector
	thrust::device_vector<float> D = H;

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//call the thrust::reduce function and time it
	cudaEventRecord(start);
	float sum = thrust::reduce(D.begin(), D.end(), 0.0, thrust::plus<float>());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print out the result of reduction and the elapsed time
	//print("%f\n%f", sum, ms);
	std::cout << sum << std::endl;
	std::cout << ms << std::endl;

	return 0;
}
