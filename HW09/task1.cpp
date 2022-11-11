#include "cluster.h"
#include <stdio.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <omp.h>

int main(int argc, char** argv){

	int n = atol(argv[1]);
	int t = atol(argv[2]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float min = 0.0, max = n;
	std::uniform_real_distribution<float> dist(min,max);
	
	//create arr
	//fill with random numbers
	float* arr = (float*) malloc (n * sizeof(float));
	for (int i = 0; i < n; i++){
		arr[i] = dist(generator);
	}

	//sort arr
	std::sort(arr, arr + n);

	//create centers
	float* centers = (float*) malloc (t * sizeof(float));
	for (int i = 1; i < t + 1; i++){
		centers[i - 1] = (2.0 * i - 1.0) * n / (2.0 * t);
	}

	//create dists
	float* dists = (float*) malloc (t * sizeof(float));
	std::fill(dists, dists + t, 0.0f);

	//call cluster and time it
	omp_set_num_threads(t);
	double startT = omp_get_wtime();
	cluster(n, t, arr, centers, dists);
	double endT = omp_get_wtime();
	double timeElapsed = (endT - startT) * 1000;

	//calculate the maximum distance in the dists array
	//get the thread number for the maximum distance in the dists array
	float max_distance = dists[0];
	int max_distance_index = 0;
	for (int i = 1; i < t; i++){
		if (dists[i] > max_distance){
			max_distance = dists[i];
			max_distance_index = i;
		}
	}

	//print the maximum distance, thread number, and elapsed time
	printf("%f\n%i\n%f\n", max_distance, max_distance_index, timeElapsed);

	return 0;
}
