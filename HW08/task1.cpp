#include <omp.h>
#include "matmul.h"
#include <stdio.h>
#include <random>

int main(int argc, char** argv){

	int n = atol(argv[1]);
	int t = atol(argv[2]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -5.0, max = 5.0; //from HW02
	std::uniform_real_distribution<float> dist_A(min,max);
	std::uniform_real_distribution<float> dist_B(min,max);

	//create A, B and C
	//fill with random numbers
	float* A = (float*) malloc (n * n * sizeof(float));
	float* B = (float*) malloc (n * n * sizeof(float));
	auto C = new float[n * n]();
	for (int i = 0; i < n * n; i++){
		A[i] = dist_A(generator);
		B[i] = dist_B(generator);
	}

	//call mmul and time it
	omp_set_num_threads(t);
	double startT = omp_get_wtime();
	mmul(A,B,C,n);
	//double timeElapsed = (omp_get_wtime() - startT) * 1000;
	double endT = omp_get_wtime();
	double timeElapsed = (endT - startT) * 1000;

	//print out the first and last elements of C + elapsed time
	printf("%f\n%f\n%lf\n", C[0], C[n * n - 1], timeElapsed);

	return 0;
}


