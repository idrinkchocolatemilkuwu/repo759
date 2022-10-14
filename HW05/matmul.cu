#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>

//reference: Lecture Note 12, slide 11-13
//reference: https://stackoverflow.com/questions/58998432/is-there-a-way-to-input-any-data-types-in-a-function-c
//reference: youtube.com/watch?v=ga2ML1uGr5o (CoffeeBeforeArch)
//reference: https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic


template <typename TYPE>
///__global__ void matmul_kernel(const TYPE *A, const TYPE *B, const TYPE *C, unsigned int n, const unsigned int block_dim);
__global__ void matmul_kernel(const TYPE *A, const TYPE *B, TYPE *C, unsigned int n, unsigned int block_dim)
{
    //shared memory for tiles of A and B
    extern __shared__ float raw_As[];
    //extern __shared__ float raw_Bs[];
    TYPE* As = reinterpret_cast<TYPE*>(raw_As);
    //TYPE* Bs = reinterpret_cast<TYPE*>(raw_Bs);

    //indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int aBegin = n * block_dim * blockIdx.y;
    unsigned int aStep = block_dim;
    unsigned int aEnd = aBegin + n - 1;
    unsigned int bBegin = block_dim * blockIdx.x;
    unsigned int bStep = block_dim * n;
    

    TYPE Csub = 0;
    for (unsigned int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    //for (unsigned int i = 0; i < n; i += blockDim.x){
	
        //load tiles
	if (a - aBegin + threadIdx.x < n && row < n){
		As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + (a - aBegin) + threadIdx.x];
	}else{
		As[threadIdx.y * blockDim.x + threadIdx.x] = 0;
	}
	if (a -aBegin + threadIdx.y < n && col < n){
		As[block_dim * block_dim + threadIdx.y * blockDim.x + threadIdx.x] = B[(a - aBegin + threadIdx.y) * n + col];
	}else{
		As[n * n + threadIdx.y * blockDim.x + threadIdx.x] = 0;
	}
	//As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * n + i + threadIdx.x];
	//Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[i * n + threadIdx.y * n + col];
        __syncthreads();

        //compute elements of C, Csub
        for (unsigned int k = 0; k < blockDim.x; ++k) {
            Csub += As[threadIdx.y * blockDim.x + k] * As[block_dim * block_dim + k * blockDim.x + threadIdx.x];
	    //printf("\nA[i]: %f x B[i]: %f = %f\n", As[threadIdx.y * block_dim + k], As[n * n + k * block_dim + threadIdx.x], Csub);
        }
	//printf("C[i]: %f\n", Csub);
        __syncthreads();
    }

    //write Csub to global memory
    //note each thread writes one element
    unsigned int c = n * block_dim * blockIdx.y + block_dim * blockIdx.x;
    //C[row * n + col] = Csub;
    if (row < n && col < n){
	    C[c + n * threadIdx.y + threadIdx.x] = Csub;
    }
}



__host__ void matmul_1(const int* A, const int* B, int* C, unsigned int n,
    unsigned int block_dim)
{
    unsigned int grid_dim = (block_dim + n - 1) / block_dim;
    matmul_kernel<int><<<dim3(grid_dim,grid_dim), dim3(block_dim, block_dim), 2*block_dim*block_dim*sizeof(int)>>>(A,B,C,n,block_dim);
    //matmul_kernel<int><<<dim3(grid_dim,grid_dim), dim3(block_dim, block_dim), 2*n*n*sizeof(int)>>>(A,B,C,n,block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float* A, const float* B, float* C, unsigned int n,
    unsigned int block_dim)
{
    unsigned int grid_dim = (block_dim + n - 1) / block_dim;
    matmul_kernel<float><<<dim3(grid_dim,grid_dim), dim3(block_dim, block_dim), 2*block_dim*block_dim*sizeof(float)>>>(A,B,C,n,block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double* A, const double* B, double* C,
    unsigned int n, unsigned int block_dim)
{
    unsigned int grid_dim = (block_dim + n - 1) / block_dim;
    matmul_kernel<double><<<dim3(grid_dim,grid_dim), dim3(block_dim, block_dim), 2*block_dim*block_dim*sizeof(double)>>>(A,B,C,n,block_dim);
    cudaDeviceSynchronize();
}
