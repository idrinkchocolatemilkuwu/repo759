#include "montecarlo.h"

int montecarlo(const size_t n, const float* x, const float* y, const float radius)
{
	int countInCircle = 0;
#pragma omp parallel for simd reduction(+:countInCircle) //for w simd
//#pragma omp parallel for reduction (+:countInCircle) //for wo simd
        for (size_t i = 0; i < n; i++){
		if (x[i] * x[i] + y[i] * y[i] < radius * radius){
			countInCircle += 1;
		}
	}
	return countInCircle;
}
