#include "matmul.h"
#include <cstddef>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <chrono>
using namespace std;
using chrono::duration;

int main(int argc, char** argv) {
	
	const size_t n = 1024;
	const float min = -5;
	const float max = 5;

	// A and B as const double
	double* A = (double*) malloc (n * n * sizeof(double));
	double* B = (double*) malloc (n * n * sizeof(double));

	for (size_t i = 0; i < n*n; i++) {
		A[i] = min + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX / (max - min)));
		B[i] = min + static_cast<float> (rand()) / (static_cast<float> (RAND_MAX / (max - min)));
	}

	// A and B as const std::vector<double>
	vector<double> vectorA;
	vector<double> vectorB;
	for (size_t i = 0; i < n * n; i++) {
		vectorA.push_back(A[i]);
		vectorB.push_back(B[i]);
	}


	cout << n << "\n\n";

	chrono::high_resolution_clock::time_point start;
	chrono::high_resolution_clock::time_point end;
	duration<double, milli> duration_sec;

	// mmul1
	auto C = new double[n * n]();
	start = chrono::high_resolution_clock::now();
	mmul1(A,B,C,n);
	end = chrono::high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, milli>>(end - start);
	cout << "mmul1 time: " << duration_sec.count() << " ms\n";
	cout << "Last element: " << C[n*n - 1] << "\n\n";
	delete[] C;

	// mmul2
	C = new double[n * n]();
	start = chrono::high_resolution_clock::now();
	mmul2(A, B, C, n);
	end = chrono::high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, milli>>(end - start);
	cout << "mmul2 time: " << duration_sec.count() << " ms\n";
	cout << "Last element: " << C[n * n - 1] << "\n\n";
	delete[] C;

	// mmul3
	C = new double[n * n]();
	start = chrono::high_resolution_clock::now();
	mmul3(A, B, C, n);
	end = chrono::high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, milli>>(end - start);
	cout << "mmul3 time: " << duration_sec.count() << " ms\n";
	cout << "Last element: " << C[n * n - 1] << "\n\n";
	delete[] C;

	// mmul4
	C = new double[n * n]();
	start = chrono::high_resolution_clock::now();
	mmul4(vectorA, vectorB, C, n);
	end = chrono::high_resolution_clock::now();
	duration_sec = std::chrono::duration_cast<duration<double, milli>>(end - start);
	cout << "mmul4 time: " << duration_sec.count() << " ms\n";
	cout << "Last element: " << C[n * n - 1] << "\n";
	delete[] C;
	
	delete[] A;
	delete[] B;
}
