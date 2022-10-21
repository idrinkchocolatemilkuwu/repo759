#include "scan.cuh"
#include <cuda.h>
#include <stdio.h>

// Performs an *inclusive scan* on the array input and writes the results to the array output.
// The scan should be computed by making calls to your kernel hillis_steele with
// threads_per_block threads per block in a 1D configuration.
// input and output are arrays of length n allocated as managed memory.
//
// Assumptions:
// - n <= threads_per_block * threads_per_block

__global__ void hillis_steele_scan(float *g_odata, float *g_idata, float *g_block, int n, int num_blocks)
{
	//reference: Lecture Note 15, page 61
	extern volatile __shared__ float temp[];

	int th_id = threadIdx.x;
	int th_len = blockDim.x;
	int bl_id = blockIdx.x;
	int index = th_len * bl_id + th_id;
	int pout = 0, pin = 1;

	if (index < n) {
		//load into shared memory
		temp[th_id] = g_idata[index];
		__syncthreads();

		//do hillis & steele scan
		for (int offset = 1; offset < th_len; offset *= 2) {
			//swap double buffer indicies
			pout = 1 - pout;
			pin = 1 - pout;
			if (offset <= th_id) {
				temp[pout * th_len + th_id] = temp[pin * th_len + th_id] + temp[pin * th_len + th_id - offset];
			}
			else {
				temp[pout * th_len + th_id] = temp[pin * th_len + th_id];
			}
			__syncthreads();
		}

		//write output
		if (pout * th_len + th_id < th_len) {
			g_odata[index] = temp[pout * n + th_id];
		}
		//for multiple blocks
		if (th_id == th_len - 1 && num_blocks != 1) {
			g_block[bl_id] = temp[pout * n + th_id];
			//printf("\n g_block[%i] = %f", bl_id, temp[pout * n + th_id]);
		}
	}
}

__global__ void add(float* g_odata, float* g_blockodata, int n)
{
	//inclusive add
	int th_id = threadIdx.x;
	int th_len = blockDim.x;
	int bl_id = blockIdx.x;
	int index = th_len * bl_id + th_id;
	if (index < n && bl_id != 0) {
		g_odata[index] += g_blockodata[blockIdx.x - 1];
	}
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block)
{
	//reference: Lecture Note 15, page 55, 61


	//none of the work should be done on host
	float* g_odata, * g_idata;
	cudaMalloc(&g_idata, n * sizeof(float));
	cudaMalloc(&g_odata, n * sizeof(float));
	//copy and paste input to g_idata
	cudaMemcpy(g_idata, input, n * sizeof(float), cudaMemcpyHostToDevice);

	//for multiple blocks do hillis & steele scan for each block
	//save the last element for each block into a seperate array g_block
	//do hillis & steele scan for g_block and output to g_blockodata
	float* g_block, *g_blockodata;
	int num_blocks = (threads_per_block + n - 1) / threads_per_block;
	cudaMalloc(&g_block, num_blocks * sizeof(float));
	cudaMalloc(&g_blockodata, num_blocks * sizeof(float));
	if (num_blocks > 1) {
		//call the hillis steele scan kernel
		hillis_steele_scan << <num_blocks, threads_per_block, 2 * threads_per_block * sizeof(float) >> > (g_odata, g_idata, g_block, n, num_blocks);
		//do scan with g_block
		hillis_steele_scan << <1, threads_per_block, 2 * threads_per_block * sizeof(float) >> > (g_blockodata, g_block, g_blockodata, num_blocks, 1);
		//add
		add << <num_blocks, threads_per_block >> > (g_odata, g_blockodata, n); 
	}
	else if (num_blocks == 1) {
		hillis_steele_scan << <num_blocks, threads_per_block, 2 * threads_per_block * sizeof(float) >> > (g_odata, g_idata, g_block, n, num_blocks);
	}
	
	//copy and paste g_odata to output
	cudaMemcpy(output, g_odata, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	
	cudaFree(g_idata); cudaFree(g_odata); cudaFree(g_block); cudaFree(g_blockodata);
}
