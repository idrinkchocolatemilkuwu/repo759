#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(){
    
    //image file path
    const char* input_image_path = "/data/data_ustv/home/ylee739/histopathpreprocessing/dataset/20069_38_2048_1462.jpg";
    //read image
    //int image_width, image_height;
    uint8_t* image;
    int image_width, image_height, bpp;
    image = stbi_load(input_image_path, &image_width, &image_height, &bpp, 3);

    //check if the loaded image width and height are correct
    //correct image width: 2048
    //correct image height: 1462
    printf("image width: %i\n", image_width);
    printf("image height: %i\n", image_height);

    //save the loaded image
    const char* output_image_path = "/data/data_ustv/home/ylee739/edge-detection-filter/output.jpg";
    stbi_write_jpg(output_image_path, image_width, image_height, 3, image, 100);
}