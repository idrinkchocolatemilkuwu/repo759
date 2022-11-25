#include "read_image.h"
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void read_image(const char* file_path, int* image, int* image_width, int* image_height){
    int bpp;
    image = stbi_load(file_path, image_width, image_height, &bpp, 3);
}