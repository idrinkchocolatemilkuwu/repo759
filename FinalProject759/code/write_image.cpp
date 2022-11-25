#include "write_image.h"
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void write_image(const char* file_path, int image_width, int image_height, int* image){
    stbi_write_jpg(file_path, image_width, image_height, image, 100);
}