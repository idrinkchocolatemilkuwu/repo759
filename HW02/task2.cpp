#include "convolution.h"
#include <cstddef>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <chrono>
using namespace std;
using chrono::duration;

int main(int argc, char** argv)
{
	size_t n = atoi(argv[1]);
	size_t m = atoi(argv[2]);
	float* image = (float*)malloc(n * n * sizeof(float));
	float* mask = (float*)malloc(m * m * sizeof(float));

	const float im_max = 10;
	const float im_min = -10;
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++){
			image[i * n + j] = im_min + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX / (im_max - im_min)));
		}
	}

	const float mask_max = 1;
	const float mask_min = -1;
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < m; j++) {
			mask[i * m + j] = mask_min + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX / (mask_max - mask_min)));
		}
	}

	float* output = (float*)malloc(n * n * sizeof(float));
	//convolve and time the convolve function
	//https://github.com/DanNegrut/759-2022/blob/main/Assignments/general/timing.md
	chrono::high_resolution_clock::time_point start;
	chrono::high_resolution_clock::time_point end;
	duration<double, milli> duration_sec;
	start = chrono::high_resolution_clock::now();
	convolve(image, output, n, mask, m);
	end = chrono::high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, milli>>(end - start);
	cout << "Scan time: " << duration_sec.count() << "ms\n";

	cout << "First element: " << output[0] << "\n";
	cout << "Last element: " << output[n - 1] << "\n";

	delete[] image;
	delete[] mask;
	delete[] output;

	return 0;
}
