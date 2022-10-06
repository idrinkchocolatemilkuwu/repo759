#include "matmul.cuh"

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n)
{
	//index for each thread
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n * n) {
		//C[ith row jth column] = sum over dot product of ith row of A and jth column of B
		//C[i * n + j] = sum over k (A[i * n + k] * B[k * n + j])
		//Note i = index / n, j = index % n, replace i and j with index and n
		for (size_t k = 0; k < n; k++) {
			C[index] += A[(index / n) * n + k] * B[k * n + (index % n)];
		}
	}
	else {
		return;
	}
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize in timing, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block)
{
	cudaMemset(C, 0, n * n * sizeof(float));
	int num_blocks = (threads_per_block + n*n - 1) / threads_per_block; 
	//call matmul_kernel
	matmul_kernel <<<num_blocks, threads_per_block >>> (A, B, C, n);
	cudaDeviceSynchronize();
}
