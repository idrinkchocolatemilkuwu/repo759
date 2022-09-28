#include <cuda.h>
#include <stdio.h>

//kernal to compute and print factorials
__global__ void PrintFactorials()
{
	int number = threadIdx.x+1;
	//initialize factorial 
	int factorial = 1;
	for (int i = 1; i <= number; i++){
		factorial *= i;
	}
	printf("%d != %d \n",number,factorial);
}

int main(){
	//kernal invocation with 8 threads
	PrintFactorials<<<1,8>>>();
	cudaDeviceSynchronize();
	return 0;
}
