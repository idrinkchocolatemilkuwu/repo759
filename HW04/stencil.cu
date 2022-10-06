#include "stencil.cuh"
#include "cuda.h"
#include "stdio.h"

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R)
{
    //index for each thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index > n + 1) {
        return;
    }
    //typecast R to avoid errors
    int R1 = R;
    //reference: nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf PAGE 8
    //int i = threadIdx.x + R1;
    int i = threadIdx.x;
    

    //declare dynamic shared memory
    extern __shared__ float shared_memory[];

    //store mask in shared memory
    float *shared_mask = shared_memory;
    if (i < 2 * R1 + 1) {
        shared_mask[i] = mask[i];
    }

    //store output in shared memory
    float *shared_output = shared_mask + 2 * R1 + 1;
    shared_output[i] = 0;

    //store image in shared memory
    //note we want to access image[i-R] and image[i+R] where max(i) = blockDim.x
    //instead of using shifted indices 
    //have shared image point at +R
    int block_width = blockDim.x;
    float *shared_image = shared_output + (block_width + R1);
    shared_image[i] = image[index];

  
    //image[i-R] = 1 when i - R < 0
    if (i < R1) {
	if (index < R1){
            shared_image[i - R1] = 1;
	}else{
	    shared_image[i - R1] = image[index - R1];
	}
	if (index + block_width < n){
	    shared_image[i + block_width] = image[index + block_width];
	}else{
	    shared_image[i + block_width] = 1;
	}
	//image [i+R] = 1 when i + R > n - 1 
	if (index + R1 > n - 1){
	    //printf("\nimage[%i]=1 stored at shared_image[%i+%i]",index,i,R1);
	    shared_image[i + R1] = 1; //edge case
	}
    }
    //image[i+R] = 1 when i + R > n - 1
    //else if (block_width - i < R1){
    //else{
    //   if (index + R1 > n - 1){
    //         shared_image[i + R1] = 1;
    //   }else{
    //   shared_image[i + R1] = image[index + R1];
    //   }
    //}


    //synchronize to make sure the matrices are loaded
    __syncthreads();

    //output[i] = sum over j (image[i+j]*mask[j+R])
    for (int j = -R1; j <= R1; j++) {
	shared_output[i] += shared_image[i + j] * shared_mask[j + R1];
	//test edge case 
	//if (index == n-1){
	    //printf("\nshared output: %f shared image: %f shared mask: %f\n", shared_output[i], shared_image[i+j], shared_mask[j+R1]);
	    //printf("\ni: %i j: %i R: %i",i,j,R1); 
	//}
    }



    //write the output elements to global memory
    output[index] = shared_output[i];

}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize in timing, that call serves the same purpose as cudaDeviceSynchronize).
__host__ void stencil(const float* image,
    const float* mask,
    float* output,
    unsigned int n,
    unsigned int R,
    unsigned int threads_per_block)
{
    //define the number of blocks
    int num_blocks = (threads_per_block + n - 1) / threads_per_block;
    //define the shared memory size where image n + 2R mask 2R+1 output n
    int shared_memory_size = (4 * R + 2 * threads_per_block + 1) * sizeof(float);
    //call the kernel
    stencil_kernel <<<num_blocks, threads_per_block, shared_memory_size >>> (image, mask, output, n, R);
    cudaDeviceSynchronize();
}
