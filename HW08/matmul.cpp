#include "matmul.h"
#include <iostream>

void mmul(const float* A, const float* B, float* C, const std::size_t n)
{
	//set C elements to 0
	for (unsigned int i = 0; i < n * n; i++){
		C[i] = 0;
	}
#pragma omp parallel for collapse(2)
	// Use mmul2 from HW02 task3 
	for (unsigned int i = 0; i < n; i++){
		for (unsigned int k = 0; k < n; k++){
			for (unsigned int j = 0; j < n; j++){
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}
