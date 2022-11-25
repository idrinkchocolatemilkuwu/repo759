#ifndef READ_IMAGE
#define READ_IMAGE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// read image
void read_image(const char* file_path, int* image, int* image_width, int* image_height);

#endif