#ifndef WRITE_IMAGE
#define WRITE_IMAGE

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// write image
void write_image(const char* file_path, int image_width, int image_height, int* image);

#endif