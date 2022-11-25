#include "read_image.h"
#include "write_image.h"

int main(){
    
    //image file path
    const char* input_image_path = "/data/data_ustv/home/ylee739/histopathpreprocessing/dataset/20069_38_2048_1462.jpg";
    //read image
    //int image_width, image_height;
    int* image, image_width, image_height;
    read_image(input_image_path, image, &image_width, &image_height);

    //check if the loaded image width and height are correct
    //correct image width: 2048
    //correct image height: 1462
    printf("image width: %i\n", image_width);
    printf("image height: %i\n", image_height);

    //save the loaded image
    const char* output_image_path = "/data/data_ustv/home/ylee739/edge-detection-filter/output.jpg";
    write_image(output_image_path, image_width, image_height, image);
}