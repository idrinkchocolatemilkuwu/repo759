#include <cuda.h>
#include <stdio.h>
#include <random>

//kernal to compute a * threadIdx + blockIdx
//reference with indexing:
//www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf, slide 39
__global__ void ComputedA(int* array_dA, int a)
{
	array_dA[threadIdx.x + blockIdx.x * blockDim.x] = a * threadIdx.x + blockIdx.x;
}

int main(){
	//initialize and allocate an array of 16 called dA
	int* dA;
	const int numElems = 16;
	cudaMalloc((void **)&dA, sizeof(int) * numElems);
	//zero out all entries
	cudaMemset(dA, 0, numElems * sizeof(int));

	//generate a randum number a in range [0,10]
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	std::uniform_int_distribution<> dist(0.,10.);
	int a = dist(generator);

	//launch kernel with 2 blocks, 8 threads/block
	ComputedA<<<2,8>>>(dA,a);
	cudaDeviceSynchronize();

	//initialize a host array called hA
	int hA[numElems];
	//bring the result back from the GPU into the host array
	cudaMemcpy(hA, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);

	//release the memory allocated on the GPU
	cudaFree(dA);
	
	//print out the 16 values stored in the host array
	for (int i = 0; i < numElems; i++){
		std::printf("%d ", hA[i]);
	}
	std::printf("\n");

	return 0;
}


	
