#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
//#include "cub/util_debug.cuh"
#include <random>

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

//reference: https://github.com/DanNegrut/759-2022/blob/main/GPU/CUB-related/deviceReduce.cu

int main(int argc, char** argv) {
    
    int n = atol(argv[1]);

    // set up random number generators
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    const float min = -1.0, max = 1.0;
    std::uniform_real_distribution<float> dist(min, max);

    // Set up host arrays
    float* h_in = new float[n];
    //float sum = 0;
    for (int i = 0; i < n; i++) {
        h_in[i] = dist(generator);
        //sum += h_in[i];
    }

    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * n));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(float) * 1));

    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // set up cuda events to time
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Do the actual reduce operation and time it
    cudaEventRecord(start);
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));

    // Check for correctness
    //printf("\t%s\n", (gpu_sum == sum ? "Test passed." : "Test falied."));
    //printf("\tSum is: %d\n", gpu_sum);

    // calculate the elapsed time in ms
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    //print out the result of reduction and the elapsed time
    printf("%f\n%f", gpu_sum, ms);

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    return 0;
}
