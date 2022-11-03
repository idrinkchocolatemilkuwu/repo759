#include "convolution.h"

float boundary_condition(const float* f, std::size_t i, std::size_t j, std::size_t n){
	if ((0 <= i && i < n) && (0 <= j && j < n)){
		return f[i * n + j];
	}
	else if ((0 <= i && i < n) || (0 <= j && j < n)){
		return 1;
	}
	else{
		return 0;
	}
}

void convolve(const float* image, float* output, std::size_t n, const float* mask, std::size_t m){
	
	for (size_t i = 0; i < n * n; i++){
		output[i] = 0;
	}

#pragma omp parallel for collapse(2)
	for (size_t x = 0; x < n; x++){
		for (size_t y = 0; y < n; y++){
			for (size_t i = 0; i < m; i++){
				for (size_t j = 0; j < m; j++){
					output[x * n + y] += mask[i * m + j] * boundary_condition(image, x + i - (m - 1) / 2, y + j - (m - 1) / 2, n);
				}
			}
		}
	}
}
