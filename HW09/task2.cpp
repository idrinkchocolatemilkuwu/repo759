#include "montecarlo.h"
#include <omp.h>
#include <stdio.h>
#include <random>

int main(int argc, char** argv){

	int n = atol(argv[1]);
	int t = atol(argv[2]);

	const float r = 1.0;
	float pi;

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = -r, max = r;
	std::uniform_real_distribution<float> dist_x(min,max);
	std::uniform_real_distribution<float> dist_y(min,max);

	//create x and y
	//fill with random numbers
	float* x = (float*) malloc (n * sizeof(float));
	float* y = (float*) malloc (n * sizeof(float));
	for (int i = 0; i < n; i++){
		x[i] = dist_x(generator);
		y[i] = dist_y(generator);
	}

	//call montecarlo and time it
	omp_set_num_threads(t);
	double startT = omp_get_wtime();
	int counts = montecarlo(n, x, y, r);
	double endT = omp_get_wtime();
	double timeElapsed = (endT - startT) * 1000;

	//compute pi
	pi = 4.0 * counts / n;

	//print out pi and elapsed time
	printf("%f\n%f\n", pi, timeElapsed);

	return 0;
}
