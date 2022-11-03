#include "convolution.h"
#include <omp.h>
#include <stdio.h>
#include <random>

int main(int argc, char** argv){

	int n = atol(argv[1]);
	int t = atol(argv[2]);

	//setup random number generators
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	const float im_min = -10.0, im_max = 10.0; //from HW02
	const float mask_min = -1.0, mask_max = 1.0; //from HW02
	std::uniform_real_distribution<float> dist_im(im_min, im_max);
	std::uniform_real_distribution<float> dist_mask(mask_min, mask_max);

	//create image, mask and output
	//fill with random numbers
	float* image = (float*) malloc (n * n * sizeof(float));
	float* mask = (float*) malloc (3 * 3 * sizeof(float));
	float* output = (float*) malloc (n * n * sizeof(float));
	for (int i = 0; i < n * n; i++){
		image[i] = dist_im(generator);
	}
	for (int i = 0; i < 3 * 3; i++){
		mask[i] = dist_mask(generator);
	}

	//call convolve and time it
	omp_set_num_threads(t);
	double startT = omp_get_wtime();
	convolve(image, output, n, mask, 3);
	double timeElapsed = (omp_get_wtime() - startT) * 1000;

	//print out the first and last elements of output + elapsed time
	printf("%f\n%f\n%lf\n", output[0], output[n * n - 1], timeElapsed);

	return 0;
}
