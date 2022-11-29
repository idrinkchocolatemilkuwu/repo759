#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//REFERENCES

//reference: https://github.com/leonnfang/Canny_filter
//reference works with PNG

//reference: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/
//           convolutionSeparable/convolutionSeparable.cu
//reference includes separable convolution kernels

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
    index = y * image_width * 3 + x * 3;
    //"luminosity method" weighs red, green, blue according to wavelength
    //instead of taking average
    mono_img[y * image_width + x] = (299 * img[index] + 587 * img[index + 1] + 114 * img[index + 2]) / 1000;

    cudaDeviceSynchronize();

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
    sum = 0;
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

__global__ void convolve(uint8_t* img, uint8_t* img_output, int img_width, int img_height, float* filter, int filter_size)
{
    //input 1D arrays of grayscale image "img" and Gaussian filter "filter"
    //where filter is filter_size x filter_size
    //convolve the image with Gaussian filter
    //store the convolved output at "img_output"

    int x, y;
    x = blockDim.x * blockIdx.x + threadIdx.x;
    y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= img_height || x >= img_width){
        return;
    }

    //do convolution
    //output[x,y] = sum over i(sum over j(image[i,j]*filter[x+i-(m-1)/2,y+j-(m-1)/2]))
    //where m is filter size 
    //note i ranges from minimum(0, x+i-(m-1)/2)
    //as i > 0 and x+i-(m-1)/2 > 0, i > (m-1)/2 -x
    //and i < image width and x+i-(m-1)/2 < m, i < 3m/2
    float sum = 0;
    for (int i = max(0, y + 1 - filter_size / 2); i < min(img_height, y + 1 + filter_size / 2); i++){
        for (int j = max(0, x + 1 - filter_size / 2); j < min(img_width, x + 1+ filter_size / 2); j++){
            sum += img[i * img_width + j] * filter[(y + i -)]
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
    
    //image file path
    const char* input_image_path = "/data/data_ustv/home/ylee739/histopathpreprocessing/dataset/20069_38_2048_1462.jpg";
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
    cudaMallocManaged((void**)&grayscale, image_width * image_height * sizeof(uint8_t));
    color_to_grayscale<<<dim3(1024,1024), dim3(2,2)>>>(device_image, grayscale_image, image_width, image_height);
    
    //compute Gaussian filter
    float* gaussian_filter;
    float gaussian_filter_size;
    float gaussian_filter_sigma = 3;
    Gaussian_filter(gaussian_filter_sigma, &gaussian_filter, &gaussian_filter_size);
    
    //convolve the image with Gaussian filter 


    //save the loaded image
    const char* output_image_path = "/data/data_ustv/home/ylee739/edge-detection-filter/output.jpg";
    stbi_write_jpg(output_image_path, image_width, image_height, 3, image, 100);
}