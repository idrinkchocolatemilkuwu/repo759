#include "scan.h"
#include <cstddef>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <chrono>
using namespace std;
using chrono::duration;

int main(int argc, char** argv)
{
	size_t n = atol(argv[1]);
	float* arr = (float*) malloc (n*sizeof(float));
	float* output = (float*) malloc (n*sizeof(float));

	//generate an array of n random floats [-1,1]
	//https://stackoverflow.com/questions/686353/random-float-number-generation
	const float min = -1;
	const float max = 1;
        //srand(time(0));
	for (size_t i = 0; i < n; i++) {
		arr[i] = min + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX / (max - min)));
	}

	//scan the array and time the scan function
	//https://github.com/DanNegrut/759-2022/blob/main/Assignments/general/timing.md
	chrono::high_resolution_clock::time_point start;
	chrono::high_resolution_clock::time_point end;
	duration<double, milli> duration_sec;
	start = chrono::high_resolution_clock::now();
	scan(arr, output, n);
	end = chrono::high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, milli>>(end - start);
	cout << "Scan time: " << duration_sec.count() << " ms\n";

	cout << "First element: " << output[0] << "\n";
	cout << "Last element: " << output[n - 1] << "\n";

	free(arr);
	free(output);
	arr = NULL;
	output = NULL;
	return 0;
}
