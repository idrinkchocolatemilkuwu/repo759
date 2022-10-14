#include "reduce.cuh"
#include <cuda.h>
#include <stdio.h>

// implements the 'first add during global load' version (Kernel 4) for the
// parallel reduction g_idata is the array to be reduced, and is available on
// the device. g_odata is the array that the reduced results will be written to,
// and is available on the device. expects a 1D configuration. uses only
// dynamically allocated shared memory.
__global__ void reduce_kernel(float* g_idata, float* g_odata, unsigned int n)
{
	//reference: lecture note 15, slide 15 - 24
	//reference: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	//reference: https://stackoverflow.com/questions/22939034/block-reduction-in-cuda

	extern __shared__ float sdata[];

	//define indices to efficiently halve the number of blocks
	unsigned int thread_id = threadIdx.x;
	//unsigned int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	//add and load
	//sdata[thread_id] = g_idata[index] + g_idata[index + blockDim.x];
	if (index < n){
		sdata[thread_id] = g_idata[index];
	}
	__syncthreads();

	//do reduction
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
	{
		if (thread_id < s)
		{
			sdata[thread_id] += sdata[thread_id + s];
		}
		__syncthreads();
	}

	//write the result to g_odata
	if (thread_id == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}

// the sum of all elements in the *input array should be written to the first
// element of the *input array. calls reduce_kernel repeatedly if needed. _No
// part_ of the sum should be computed on host. *input is an array of length N
// in device memory. *output is an array of length = (number of blocks needed
// for the first call of the reduce_kernel) in device memory. configures the
// kernel calls using threads_per_block threads per block. the function should
// end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float** input, float** output, unsigned int N,
	unsigned int threads_per_block)
{
	float *g_idata, *g_odata;
	cudaMallocManaged(&g_idata, N * sizeof(float));
	cudaMallocManaged(&g_odata, (N + threads_per_block - 1) / threads_per_block * sizeof(float));
	cudaMemcpy(g_idata, (*input), N * sizeof(float), cudaMemcpyHostToDevice);
	//g_idata = *input;
	//g_odata = *output;

	//call kernel
	//sum should be computed not on host
	for (int i = N; i != 1; i = (i + threads_per_block - 1) / threads_per_block){
		int num_block = (i + threads_per_block - 1) / threads_per_block;
		//reduce_kernel <<<num_block, threads_per_block, threads_per_block* sizeof(float) >>> (*input, *output, i);
		reduce_kernel<<<num_block, threads_per_block, threads_per_block * sizeof(float)>>>(g_idata, g_odata, i);
		cudaMemcpy(g_idata,g_odata,num_block * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	//cudaDeviceSynchronize();

	//write the sum to first element of *input
	float sum = g_odata[0];
	//printf("sum: %f", sum);
	*input[0] = sum;

	cudaFree(g_idata);
	cudaFree(g_odata);

	cudaDeviceSynchronize();
}
