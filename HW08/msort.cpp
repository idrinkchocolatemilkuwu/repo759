#include "msort.h"
#include <algorithm>
#include <iostream>

//reference: stackoverflow.com/questions/13811114/parallel-merge-sort-in-openmp
//reference: geeksforgeeks.org/insertion-sort/
//reference: stackoverflow.com/questions/12030683/implementing-merge-sort-in-c

//serial sort algorithm
/*
void insertionSort(int *arr, const std::size_t n)
{
	int key, j;
	for (size_t i = 1; i < n; i++){
		key = arr[i];
		j = i - 1;
		while ( j >= 0 && arr[j] > key)
		{
			arr[j + 1] = arr[j];
			j -= 1;
		}
		arr[j + 1] = key;
	}
	//printf("performing insertion sort");

}*/

void insertionSort(int* arr, const std::size_t n)
{
	//reference: geeksforgeeks.org/insertion-sort-using-c-stl/
	for (auto ptr = arr; ptr != arr + n; ptr++){
		auto const insertion_point = std::upper_bound(arr, ptr, *ptr);
		std::rotate(insertion_point, ptr, ptr + 1);
	}
	return;
}

//openmp parallel sort
void mergesort_parallel_omp(int* arr, const std::size_t n, const std::size_t threshold, int number_threads)
{
	//threshold is the lower limit of array size
	//where the function starts making parallel recursive calls
	if (n < threshold){
		//use a serial sort algorithm
		insertionSort(arr, n);
		return;
	}

	//if n >= threshold
	//start making parallel recursive calls
	if (number_threads == 1){
		//insertionSort(arr, n);
		//return;
		mergesort_parallel_omp(arr, n/2, threshold, number_threads);
		mergesort_parallel_omp(arr + n/2, n - n/2, threshold, number_threads);
	}
	else if (number_threads > 1){
#pragma omp task
		mergesort_parallel_omp(arr, n/2, threshold, number_threads / 2);
#pragma omp task
		mergesort_parallel_omp(arr + n/2, n - n/2, threshold, number_threads - number_threads / 2);
#pragma omp taskwait
	}
		std::inplace_merge(arr, arr + n/2, arr + n);
		return;
}

void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
#pragma omp parallel
#pragma omp single
	mergesort_parallel_omp(arr, n, threshold, omp_get_num_threads());
}
