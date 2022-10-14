#include "reduce.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

int main(int argc, char **argv){

	unsigned int N = atol(argv[1]);
	unsigned int threads_per_block = atol(argv[2]);
	unsigned int num_blocks = (threads_per_block + N - 1) / threads_per_block;

	//declare input, output
	float **input, **output;
	//declare it, ot
	//we will have input and output point to it and ot
	//where it is an array of length N
	//and ot is an array of length num_blocks

	//memory try 1
	//float *it = (float *)malloc(sizeof(float) * N);
	//float *ot = (float *)malloc(sizeof(float) * num_blocks);

	//memory try 2
	//auto it = new float[N]();

	//memory try 3
	float *it, *ot;
	cudaMallocManaged((void**)&(it), sizeof(float) * N);
	cudaMallocManaged((void**)&(ot), sizeof(float) * num_blocks);
	input = &it;
	output = &ot;

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -1.0, max = 1.0;
	std::uniform_real_distribution<float> dist(min,max);

	//fill it with random numbers
	//float sum = 0;
	for (unsigned int i = 0; i <  N; i++){
		it[i] = dist(generator);
		//sum += it[i];
	}

	//set up cuda events to time
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//call the reduce function and time it
	cudaEventRecord(start);
	reduce(input, output, N, threads_per_block);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//calculate the elapsed time in ms
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	//print the resulting sum and time
	printf("%f\n%f", it[0],ms);
	//if (it[0] != sum){printf("\nerror ot: %f sum: %f\n", it[0], sum);}

	//deallocate
	cudaFree(it); cudaFree(ot);
	cudaFree(input); cudaFree(output);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
	
}
