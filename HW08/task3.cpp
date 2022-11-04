#include "msort.h"
#include <omp.h>
#include <stdio.h>
#include <random>

int main(int argc, char** argv){

	int n = atol(argv[1]);
	int t = atol(argv[2]);
	size_t ts = atol(argv[3]);

	//set up random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const int min = -1000, max = 1000;
	std::uniform_int_distribution<int> dist(min,max);
	
	//create arr
	//fill with random numbers
	int* arr = (int*) malloc (n * sizeof(int));
	for (int i = 0; i < n; i++){
	       arr[i] = dist(generator);
	}
	
	//call msort and time it
	omp_set_num_threads(t);
	omp_set_nested(1);
	double startT = omp_get_wtime();
	msort(arr, n, ts);
	double timeElapsed = (omp_get_wtime() - startT) * 1000;

	//print out the first and last elements of arr + elapsed time
	printf("%i\n%i\n%lf\n", arr[0], arr[n-1], timeElapsed);
	
	return 0;
}	
