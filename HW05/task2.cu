#include "matmul.cuh"
#include <cuda.h>
#include <stdio.h>
#include <random>

int main(int argc, char **argv){
	unsigned int n = atol(argv[1]);
	unsigned int block_dim = atol(argv[2]);

	//matmul_1
	int *A_in, *B_in, *C_in;
	cudaMallocManaged((void **)&A_in, n * n * sizeof(int));
	cudaMallocManaged((void **)&B_in, n * n * sizeof(int));
	cudaMallocManaged((void **)&C_in, n * n * sizeof(int));
	//A_in will store 0,1,...,n * n - 1
	//B_in will be identity matrix
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < n; ++j){
			A_in[i * n + j] = int(i * n + j);
			if (i == j){
				B_in[i * n + j] = 1;
			}else{
				B_in[i * n + j] = 0;
			}
		}
	}
	//set up cuda events to time
	cudaEvent_t start_1;
	cudaEvent_t stop_1;
	cudaEventCreate(&start_1);
	cudaEventCreate(&stop_1);
	//call matmul_1 and time the function
	cudaEventRecord(start_1);
	matmul_1(A_in,B_in,C_in,n,block_dim);
	cudaEventRecord(stop_1);
	cudaEventSynchronize(stop_1);
	//calculate the elasped time in ms
	float ms_1;
	cudaEventElapsedTime(&ms_1, start_1, stop_1);
	//print out the first element, last element and elapsed time
	printf("%i\n%i\n%f\n",C_in[0],C_in[n * n - 1],ms_1);
	//deallocate
	cudaEventDestroy(start_1);
	cudaEventDestroy(stop_1);
	cudaFree(A_in);
	cudaFree(B_in);
	cudaFree(C_in);

	
	//matmul_2
	float *A_fl, *B_fl, *C_fl;
	cudaMallocManaged((void **)&A_fl, n * n * sizeof(float));
	cudaMallocManaged((void **)&B_fl, n * n * sizeof(float));
	cudaMallocManaged((void **)&C_fl, n * n * sizeof(float));
	//A_fl will store 0,1,...,n * n - 1
	//B_fl will be identity matrix
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < n; ++j){
			A_fl[i * n + j] = float(i * n + j);
			if (i == j){
				B_fl[i * n + j] = 1;
			}else{
				B_fl[i * n + j] = 0;
			}
		}
	}
	//set up cuda events to time
	cudaEvent_t start_2;
	cudaEvent_t stop_2;
	cudaEventCreate(&start_2);
	cudaEventCreate(&stop_2);
	//call matmul_2 and time the function
	cudaEventRecord(start_2);
	matmul_2(A_fl,B_fl,C_fl,n,block_dim);
	cudaEventRecord(stop_2);
	cudaEventSynchronize(stop_2);
	//calculate the elapsed time in ms
	float ms_2;
	cudaEventElapsedTime(&ms_2, start_2, stop_2);
	//print out the first element, last element and elapsed time
	printf("%f\n%f\n%f\n",C_fl[0],C_fl[n * n - 1],ms_2);
	//deallocate
	cudaEventDestroy(start_2);
	cudaEventDestroy(stop_2);
	cudaFree(A_fl);
	cudaFree(B_fl);
	cudaFree(C_fl);


	//matmul_3
	double *A_db, *B_db, *C_db;
	cudaMallocManaged((void **)&A_db, n * n * sizeof(double));
	cudaMallocManaged((void **)&B_db, n * n * sizeof(double));
	cudaMallocManaged((void **)&C_db, n * n * sizeof(double));
	//A_db will store 0,1,...,n * n - 1
	//B_db will be identity matrix
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < n; ++j){
			A_db[i * n + j] = double(i * n + j);
			if (i == j){
				B_db[i * n + j] = 1;
			}else{
				B_db[i * n + j] = 0;
			}
		}
	}
	//set up cuda events to time
	cudaEvent_t start_3;
	cudaEvent_t stop_3;
	cudaEventCreate(&start_3);
	cudaEventCreate(&stop_3);
	//call matmul_3 and time the function
	cudaEventRecord(start_3);
	matmul_3(A_db,B_db,C_db,n,block_dim);
	cudaEventRecord(stop_3);
	cudaEventSynchronize(stop_3);
	//calculate the elapsed time in ms
	float ms_3;
	cudaEventElapsedTime(&ms_3, start_3, stop_3);
	//print out the first element, last element and elapsed time
	printf("%f\n%f\n%f",C_db[0],C_db[n * n - 1],ms_3);
	//deallocate
	cudaEventDestroy(start_3);
	cudaEventDestroy(stop_3);
	cudaFree(A_db);
	cudaFree(B_db);
	cudaFree(C_db);

	return 0;
}

