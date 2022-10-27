#include "count.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

// Find the unique integers in the array d_in,
// store these integers in values array in ascending order,
// store the occurrences of these integers in counts array.
// values and counts should have equal length.
// Example:
// d_in = [3, 5, 1, 2, 3, 1]
// Expected output:
// values = [1, 2, 3, 5]
// counts = [2, 1, 2, 1]
void count(const thrust::device_vector<int>& d_in,
    thrust::device_vector<int>& values,
    thrust::device_vector<int>& counts)
{
    //reference: https://thrust.github.io/doc/group__reductions_gad5623f203f9b3fdcab72481c3913f0e0.html
    
    //copy d_in to d_values
    thrust::device_vector<int> d_values(d_in.size());
    thrust::copy(thrust::device, d_in.begin(), d_in.end(), d_values.begin());

    //sort values in ascending order
    thrust::sort(d_values.begin(), d_values.end());

    //fill d_counts with 1 so we can +1 count for each key
    thrust::device_vector<int> d_counts(d_in.size());
    thrust::fill(d_counts.begin(), d_counts.end(), 1);

    //find the unique integers and counts
    auto result = thrust::reduce_by_key(d_values.begin(), d_values.end(), d_counts.begin(), values.begin(), counts.begin());
    values.resize(result.first - values.begin());
    counts.resize(result.second - counts.begin());
}