#include "mmul.h"

// Uses a single cuBLAS call to perform the operation C := A B + C
// handle is a handle to an open cuBLAS instance
// A, B, and C are matrices with n rows and n columns stored in column-major
// NOTE: The cuBLAS call should be followed by a call to cudaDeviceSynchronize() for timing purposes

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n)
{
	//reference: https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf page 82-83
	//function performs C= alpha AB + beta C where alpha and beta are scalars

	float alpha = 1.f;
	float beta = 1.f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
	cudaDeviceSynchronize();
}
