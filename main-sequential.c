#include <stdlib.h>
#include <math.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "helpers/stb_image.h"
#include "helpers/stb_image_write.h"

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1

unsigned long findMin(unsigned long* cdf){
    
    unsigned long min = 0;
    // grem skozi CDF dokler ne najdem prvi nenicelni element ali pridem do konca
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}

unsigned char Scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize){
    
    float scale;
    
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    
    scale = round(scale * (float)(GRAYLEVELS-1));
    
    return (int)scale;
}


void Equalize(unsigned char * image_in, unsigned char * image_out, int width, int height, unsigned long* cdf){
     
    unsigned long imageSize = width * height;
    
    unsigned long cdfmin = findMin(cdf);
    
    //Equalize: namig: blok niti naj si CDF naloži v skupni pomnilnik
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            image_out[(i*width + j)] = Scale(cdf[image_in[i*width + j]], cdfmin, imageSize);
        }
    }
}

/*
NAMIG: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda 
*/
void CalculateCDF(unsigned long* histogram, unsigned long* cdf){
    
    // clear cdf:
    for (int i=0; i<GRAYLEVELS; i++) {
        cdf[i] = 0;
    }
    
    // calculate cdf from histogram
    cdf[0] = histogram[0];
    for (int i=1; i<GRAYLEVELS; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
}

void CalculateHistogram(unsigned char* image, int width, int height, unsigned long* histogram){
    
    //Clear histogram:
    for (int i=0; i<GRAYLEVELS; i++) {
        histogram[i] = 0;
    }
    
    //Calculate histogram: namig: Cuda by Example, poglavje 9, str. 179
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            histogram[image[i*width + j]]++;
        }
    }
}

float run(unsigned char* imageIn, int width, int height, int cpp) {
    clock_t begin = clock();

    //Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned long));
    unsigned long *histogram= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));
    unsigned long *CDF= (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));

    CalculateHistogram(imageIn, width, height, histogram);
    CalculateCDF(histogram, CDF);
    Equalize(imageIn, imageOut, width, height, CDF);

    clock_t end = clock();
    float elapsedTime = (float)(end - begin) / CLOCKS_PER_SEC * 1000;

    stbi_write_jpg("images-output/output-sequential-c.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

    free(imageOut);
    free(histogram);
    free(CDF);

    return elapsedTime;
}

int main(int argc, char** argv){

    // Read image from file
    int width, height, cpp;
    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *imageIn = stbi_load(argv[1], &width, &height, &cpp, DESIRED_NCHANNELS);
    if(imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }

    float timeSum = 0;
    const int REPETITIONS = atoi(argv[2]);
    for (int i = 0; i < REPETITIONS; i++)
    {
        timeSum += run(imageIn, width, height, cpp);
    }
    printf("Average time: %.2f ms\n", timeSum/REPETITIONS);
    
    free(imageIn);

    return 0;
}