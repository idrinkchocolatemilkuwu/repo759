#include <omp.h>
#include <stdio.h>

int GetFactorial(int n)
{
	int factorial = 1;
	for (int i = 1; i <= n; i++) {
		factorial *= i;
	}
	return factorial;
}

int main() {
	//reference: Lecture Note 20 page 31, 34, 38
	
	//launch four OpenMP threads
	int num_threads = 4;
	omp_set_num_threads(num_threads);

	//print out the number of threads launched
	printf("Number of threads: %i\n", num_threads);

	//print out the thread number
#pragma omp parallel
	{
		int myId = omp_get_thread_num();
		printf("I am thread No. %i\n", myId);
	}

	//compute and print out factorial
#pragma omp parallel for
	for (int i = 1; i <= 8; i++) {
		printf("%i!=%i\n", i, GetFactorial(i));
	}

	return 0;
}