#include "convolution.h"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
	const size_t n = 4;
	const size_t m = 3;
	float image[n * n] = { 1,3,4,8,6,5,2,4,3,4,6,8,1,4,5,2 };
	float mask[m * m] = { 0,0,1,0,1,0,1,0,0 };
	float output[n * n];
	convolve(image, output, n, mask, m);
	for (size_t i = 0; i < n*n; i++){
		cout << output[i] << "  ";
	}
	return 0;
}
