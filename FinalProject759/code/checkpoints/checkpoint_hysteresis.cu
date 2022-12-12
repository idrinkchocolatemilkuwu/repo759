#include <stdint.h>
#include <cmath>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//REFERENCES

//reference: https://en.wikipedia.org/wiki/Canny_edge_detector
//reference includes the Canny edge detection algorithm

//reference: https://github.com/leonnfang/Canny_filter
//reference includes Canny edge detector implemented in CUDA that works with PNG

//reference: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/
//           convolutionSeparable/convolutionSeparable.cu
//reference includes separable convolution kernels

//reference: https://en.wikipedia.org/wiki/Canny_edge_detector
//reference includes the Sobel operator

///////////////////////////////////////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void color_to_grayscale(uint8_t* img, uint8_t* mono_img, int img_width, int img_height)
{
    //input 1D array of color image "img"
    //where img consists of R G B values of img_width * image_height pixels
    //store 1D array of grayscale image of img at "mono_img"

    int x, y;
    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= img_height || x >= img_width){
        return;
    }

    int index;
    index = y * img_width * 3 + x * 3;
    //"luminosity method" weighs red, green, blue according to wavelength
    //instead of taking average
    mono_img[y * img_width + x] = (299 * img[index] + 587 * img[index + 1] + 114 * img[index + 2]) / 1000;

    //cudaDeviceSynchronize();

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void Gaussian_filter(float sigma, float** computed_filter, float* computed_filter_size)
{
    //compute Gaussian filter
    //input sigma (standard deviation) for the filter
    //store the computed filter at "computed_filter"
    //store the computed filter size at "computed_filter_size"

    float* filter;
    //matrix of size 6 sigma x 6 sigma is typically used
    float filter_size = 6 * sigma;
    float filter_center = (filter_size - 1) / 2;
    cudaMallocManaged((void**)&filter, filter_size * sizeof(float));
    
    //compute Gaussian filter
    float sum = 0;
    for (int i = 0; i < filter_size; i++){
        filter[i] = exp(-pow((i - filter_center) / sigma, 2) / 2) / (sigma * sqrt(2 * M_PI));
        sum += filter[i];
    }

    //normalize filter
    for (int i = 0; i < filter_size; i++){
        filter[i] /= sum;
    }

    *computed_filter = filter;
    *computed_filter_size = filter_size;

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ uint8_t boundary_condition(uint8_t* img, int img_width, int img_height, float* filter, int filter_size, int i, int j)
{
    //boundary condition for convolution
    //try boundary condition from HW02

    if ((0 <= i && i < img_height) && (0 <= j && j < img_width)){
        return img[i * img_width + j];
    }
    else if ((0 <= i && i < img_height) || (0 <= j && j < img_width)){
        return 1;
    }
    else{
        return 0;
    }
}

__host__ void convolve(uint8_t* img, uint8_t* img_output, int img_width, int img_height, float* filter, int filter_size)
{
    //input 1D arrays of grayscale image "img" and Gaussian filter "filter"
    //where filter is filter_size x filter_size
    //convolve the image with Gaussian filter
    //store the convolved output at "img_output"

    //set img_output to 0
    for (int i = 0; i < img_width * img_height; i++){
        img_output[i] = 0;
    }

    //do convolution
    #pragma omp parallel for collapse(2)
    for (int x = 0; x < img_height; x++){
        for (int y = 0; y < img_width; y++){
            for (int i = 0; i < filter_size; i++){
                for (int j = 0; j < filter_size; j++){
                    img_output[x * img_width + y] += filter[i * filter_size + j] * boundary_condition(img, img_width, img_height, filter, filter_size, x + i - (filter_size - 1) / 2, y + j - (filter_size - 1) / 2);
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void compute_magnitude_and_angle(uint8_t* g_x, uint8_t* g_y, uint8_t* g_abs, float* g_theta, int width, int height)
{
    //input 1D arrays of the gradient "g_x" and "g_y"
    //compute the magnitude of the gradient = square root(g_x^2 + g_y^2)
    //store the magnitude at "g_abs"
    //compute the gradient's direction = atan2(g_x, g_y)
    //store the direction at "g_theta"

    #pragma omp parallel for
    for (int i = 0; i < width * height; i++){
        //compute magnitude
        g_abs[i] = sqrt(g_x[i] * g_x[i] + g_y[i] * g_y[i]);
        //try setting a threshold
        /*uint8_t tmp = abs(g_x[i]) + abs(g_y[i]);
        if (tmp > 255){tmp = 255;}
        else if (tmp < 150){tmp = 0;}
        g_abs[i] = tmp;*/

        //compute angle
        //note atan2 maps -180 to 180.
        //we can use (theta + 180) % 180 to have theta range from 0 to 180
        float tmp_theta = atan2f(g_y[i], g_x[i]);
        tmp_theta *= 180 / M_PI;
        tmp_theta += 180;
        tmp_theta = fmod(tmp_theta, 180);
        //printf("converting angle from radians to degree: %f\n", tmp_theta);

        //also note theta in range (22.5, 67.5) should map to 45
        //     theta in range (67.5, 112.5) should map to 90
        //     ...
        //theta should map to ((theta + 22.5) / 45) * 45
        tmp_theta += 22.5;
        tmp_theta = std::floor(tmp_theta / 45) * 45;
        //printf("mapping the converted angles to: %f\n", tmp_theta);

        //combining the above,
        //theta should map to (((theta + 180 + 22.5) / 45) * 45) % 180 
        //g_theta[i] = fmod(std::floor((atan2f(g_y[i], g_x[i]) + 9 * M_PI / 80 ) / (M_PI / 4)) * (M_PI / 4), M_PI);
    }
}

__host__ void intensity_gradient(uint8_t* img, int img_width, int img_height, uint8_t* g_x, uint8_t* g_y, uint8_t* gradient, float* gradient_dir)
{
    //input 1D array of image "img"
    //compute the intensity gradient of the image using the Sobel operator
    //store the gradient_x at "g_x" 
    //store the gradient_y at "g_y"
    //store the gradient at "gradient"
    //store the gradient direction at "gradient_dir"

    //compute the intensity gradient
    //g_x = [ 1  0 -1]
    //      [ 2  0 -2] * img
    //      [ 1  0 -1]
    //g_y = [ 1  2  1]
    //      [ 0  0  0] * img
    //      [-1 -2 -1]
    float filter_x[9] = {1,0,-1,2,0,-2,1,0,-1};
    float filter_y[9] = {1,2,1,0,0,0,-1,-2,-1};
    convolve(img, g_x, img_width, img_height, filter_x, 3);
    convolve(img, g_y, img_width, img_height, filter_y, 3);
    compute_magnitude_and_angle(g_x, g_y, gradient, gradient_dir, img_width, img_height);
}

__global__ void intensity_gradient_kernel(uint8_t* img, int img_width, int img_height, uint8_t* g_x, uint8_t* g_y, uint8_t* gradient, float* gradient_dir)
{
    //try kernel from reference
    int x, y;
    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= img_height || x >= img_width){
        return;
    }

    g_x[y * img_width + x] = img[(y - 1) * img_width + (x - 1)] - img[(y - 1) * img_width + (x + 1)]
                           + 2 * img[y * img_width + (x - 1)] - 2 * img[y * img_width + (x + 1)]
                           + img[(y + 1) * img_width + (x - 1)] - img[(y + 1) * img_width + (x + 1)];
    g_y[y * img_width + x] = img[(y - 1) * img_width + (x - 1)] + 2 * img[(y - 1) * img_width + x] + img[(y - 1) * img_width +(x + 1)] 
                           - img[(y + 1) * img_width + (x - 1)] - 2 * img[(y + 1) * img_width + x] - img[(y + 1) * img_width +(x + 1)];
    
    gradient[y * img_width + x] = sqrtf(g_x[y * img_width + x] * g_x[y * img_width + x] + g_y[y * img_width + x] * g_y[y * img_width + x]);

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void apply_lower_bound_to_gradient(uint8_t* gradient, float* gradient_dir, uint8_t* updated_gradient, int img_width, int img_height)
{
    //perform the "edge thinning" technique
    //input the gradient "gradient" and gradient direction "gradient_dir"
    //store the updated gradient at "updated gradient"

    int x, y;
    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= img_height || x >= img_width){
        return;
    }

    //for each pixel check the gradient direction and the neighboring gradient magnitude values
    float neighbor_gradient_1, neighbor_gradient_2;
    bool gradient_value_as_expected = false;
    if (gradient_dir[y * img_width + x] == 0){
        //the edge is in north-south direction
        //the pixel will be considered to be on edge
        //if its gradient magnitude is greater than 
        //the neighboring gradient magnitudes in the east-west direction
        neighbor_gradient_1 = gradient[y * img_width + x + 1];
        neighbor_gradient_2 = gradient[y * img_width + x - 1];
        gradient_value_as_expected = true;
    }
    if (gradient_dir[y * img_width + x] == 90){
        //the edge is in east-west direction
        //the pixel will be considered to be on edge
        //if its gradient magnitude is greater than 
        //the neighboring gradient magnitudes in the north-south direction
        neighbor_gradient_1 = gradient[(y + 1) * img_width + x];
        neighbor_gradient_2 = gradient[(y - 1) * img_width + x];
        gradient_value_as_expected = true;
    }
    if (gradient_dir[y * img_width + x] == 135){
        //the edge is in northeast-southwest direction
        //the pixel will be considered to be on edge
        //if its gradient magnitude is greater than 
        //the neighboring gradient magnitudes in the northwest-southeast direction
        neighbor_gradient_1 = gradient[(y + 1) * img_width + (x - 1)];
        neighbor_gradient_2 = gradient[(y - 1) * img_width + (x + 1)];
        gradient_value_as_expected = true;
    }
    if (gradient_dir[y * img_width + x] == 45){
        //the edge is in northwest-southeast direction
        //the pixel will be considered to be on edge
        //if its gradient magnitude is greater than 
        //the neighboring gradient magnitudes in the northeast-southwest direction
        neighbor_gradient_1 = gradient[(y + 1) * img_width + (x + 1)];
        neighbor_gradient_2 = gradient[(y - 1) * img_width + (x - 1)];
        gradient_value_as_expected = true;
    }

    /*if (not(gradient_value_as_expected)){
        //if gradient angle is not 0 or 45 or ... 135
        //print
        printf("check intensity gradient\n");
        printf("current gradient angle is: %f\n", gradient_dir[y * img_width + x]);
    }*/

    //compare the gradient with the neighboring gradients
    if (gradient[y * img_width + x] < neighbor_gradient_1 || gradient[y * img_width + x] < neighbor_gradient_2){
        updated_gradient[y * img_width + x] = 0;
    }
    else{
        updated_gradient[y * img_width + x] = gradient[y * img_width + x];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//extern uint8_t weak_edge_pixel_value = 190;
//extern uint8_t strong_edge_pixel_value = 255;
#define weak_edge_pixel_value 190
#define strong_edge_pixel_value 255

__global__ void apply_double_threshold_to_gradient(uint8_t* gradient, int img_width, int img_height, uint8_t lower_threshold, uint8_t upper_threshold)
{
    //apply double thresholds to image gradient
    //input the gradient at "gradient" and threshold values at "lower_threshold" and "upper_threshold"
    //store the updated gradient at "gradient"

    int x, y;
    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= img_height || x >= img_width){
        return;
    }

    int index;
    index = y * img_width + x;

    //if the gradient value is lower than the lower threshold
    //suppress
    if (gradient[index] < lower_threshold){
        gradient[index] = 0;
    }
    //if the gradient value is between the lower threshold and the upper threshold
    //mark it as a weak edge
    else if (gradient[index] < upper_threshold){
        gradient[index] = weak_edge_pixel_value;
    }
    //if the gradient value is higher than the upper threshold
    //mark it as a strong edge
    else{
        gradient[index] = strong_edge_pixel_value;
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void mark_weak_edge_as_strong_edge(uint8_t* gradient, int x, int y, int img_width, int img_height)
{
    //input gradient "gradient"
    //check if "gradient"["y" * "img_width" + "x"] is marked as weak edge
    //mark it as strong edge and update the "gradient"

    if (0 <= x and x < img_width and 0 <= y and y < img_height){
        //out of bounds
        return;
    }
    else{
        int index = y * img_width + x;
        if (gradient[index] == weak_edge_pixel_value){
            gradient[index] = strong_edge_pixel_value;
        }
    }
}

__host__ void hysteresis(uint8_t* gradient, int img_width, int img_height)
{
    //input gradient "gradient"
    //apply hysteresis
    //store the updated gradient at "gradient"

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){
            int index = y * img_width + x;

            //if gradient is considered on edge
            //check the neighboring gradients
            //and consider the ones marked as "weak edge" from earlier to be on edge 
            if (gradient[index] == strong_edge_pixel_value){
                mark_weak_edge_as_strong_edge(gradient, x - 1, y - 1, img_width, img_height);
                mark_weak_edge_as_strong_edge(gradient, x, y - 1, img_width, img_height);
                mark_weak_edge_as_strong_edge(gradient, x + 1, y - 1, img_width, img_height);

                mark_weak_edge_as_strong_edge(gradient, x - 1, y, img_width, img_height);
                //mark_weak_edge_as_strong_edge(gradient, x, y, img_width, img_height);
                mark_weak_edge_as_strong_edge(gradient, x + 1, y, img_width, img_height);

                mark_weak_edge_as_strong_edge(gradient, x - 1, y + 1, img_width, img_height);
                mark_weak_edge_as_strong_edge(gradient, x, y + 1, img_width, img_height);
                mark_weak_edge_as_strong_edge(gradient, x + 1, y + 1, img_width, img_height);
            }

        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void cleanup_image(uint8_t* gradient, int img_width, int img_height)
{
    //clean up the image
    //input gradient at "gradient"
    //set the gradient value to 0 if gradient is not marked as on edge

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){
            int index = y * img_width + x;
            if (gradient[index] != strong_edge_pixel_value){
                gradient[index] = 0;
            }
        }
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
    
    //image file path
    //const char* input_image_path = "/data/data_ustv/home/ylee739/histopathpreprocessing/dataset/20069_38_2048_1462.jpg";
    const char* input_image_path = "/data/data_ustv/home/ylee739/edge-detection-filter/Large_Scaled_Forest_Lizard.jpg";
    
    //read image
    //int image_width, image_height;
    uint8_t* image;
    int image_width, image_height, bpp;
    image = stbi_load(input_image_path, &image_width, &image_height, &bpp, 3);

    //image points to the pixel data where
    //pixel data consists of image_height scanlines of image_width pixels
    //each pixel consists of 3 components

    //copy image to device
    uint8_t* device_image;
    cudaMallocManaged((void**)&device_image, image_width * image_height * 3 * sizeof(uint8_t));
    cudaMemcpy(device_image, image, image_width * image_height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    //change image to grayscale
    uint8_t* grayscale_image;
    cudaMallocManaged((void**)&grayscale_image, image_width * image_height * sizeof(uint8_t));
    color_to_grayscale<<<dim3(1024,1024), dim3(4,4)>>>(device_image, grayscale_image, image_width, image_height);
    
    //compute Gaussian filter
    float* gaussian_filter;
    float gaussian_filter_size;
    float gaussian_filter_sigma = 3; //3 //1.4 //9
    Gaussian_filter(gaussian_filter_sigma, &gaussian_filter, &gaussian_filter_size);
    
    //convolve the image with Gaussian filter 
    uint8_t* convolved_image;
    cudaMallocManaged((void**)&convolved_image, image_width * image_height * sizeof(uint8_t));
    convolve(grayscale_image, convolved_image, image_width, image_height, gaussian_filter, gaussian_filter_size);

    //compute the intensity gradient
    uint8_t* gradient_x; uint8_t* gradient_y; 
    cudaMallocManaged((void**)&gradient_x, image_width * image_height * sizeof(uint8_t));
    cudaMallocManaged((void**)&gradient_y, image_width * image_height * sizeof(uint8_t));
    uint8_t* image_gradient; float* image_gradient_dir;
    cudaMallocManaged((void**)&image_gradient, image_width * image_height * sizeof(uint8_t));
    cudaMallocManaged((void**)&image_gradient_dir, image_width * image_height * sizeof(float));
    intensity_gradient(convolved_image, image_width, image_height, gradient_x, gradient_y, image_gradient, image_gradient_dir);
    //intensity_gradient_kernel<<<dim3(1024,1024), dim3(4,4)>>>(convolved_image, image_width, image_height, gradient_x, gradient_y, image_gradient, image_gradient_dir);

    //perform edge thinning
    uint8_t* gradient_after_edge_thinning;
    cudaMallocManaged((void**)&gradient_after_edge_thinning, image_width * image_height * sizeof(uint8_t));
    apply_lower_bound_to_gradient<<<dim3(1024,1024), dim3(4,4)>>>(image_gradient, image_gradient_dir, gradient_after_edge_thinning, image_width, image_height);

    //set threshold values for double threshold
    uint8_t lower_threshold_value = 100;
    uint8_t upper_threshold_value = 150;
    //apply double threshold
    apply_double_threshold_to_gradient<<<dim3(1024,1024), dim3(4,4)>>>(gradient_after_edge_thinning, image_width, image_height, lower_threshold_value, upper_threshold_value);

    //apply hysteresis
    hysteresis(gradient_after_edge_thinning, image_width, image_height);

    //clean up the image
    cleanup_image(gradient_after_edge_thinning, image_width, image_height);
    
    //saves and check
    const char* output_image_path = "/data/data_ustv/home/ylee739/edge-detection-filter/output.jpg";
    stbi_write_jpg(output_image_path, image_width, image_height, 1, gradient_after_edge_thinning, 100);
    //stbi_write_jpg(output_image_path, image_width, image_height, 3, image, 100);
}