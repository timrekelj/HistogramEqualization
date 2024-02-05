#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "helpers/stb_image.h"
#include "helpers/stb_image_write.h"

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1

__global__ void histogram_kernel(unsigned char* image, int width, int height, unsigned int* histogram) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    __shared__ unsigned int localHistogram[GRAYLEVELS];

    localHistogram[blockDim.x * ly + lx] = 0;
    __syncthreads();

    if (x < width && y < height) {
        atomicAdd(&localHistogram[image[y * width + x]], 1);
    }

    __syncthreads();

    atomicAdd(&(histogram[blockDim.x * ly + lx]), localHistogram[blockDim.x * ly + lx]);
}

__global__ void CDF_kernel(unsigned int* histogram, unsigned int* cdf) {
    __shared__ unsigned int temp[GRAYLEVELS * 2];
    int tid = threadIdx.x;

    int pout = 0, pin = 1;

    temp[tid] = histogram[tid];

    __syncthreads();

    for (int offset = 1; offset < GRAYLEVELS; offset <<= 1) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (tid >= offset)
            temp[pout*GRAYLEVELS+tid] = temp[pin*GRAYLEVELS+tid] + temp[pin*GRAYLEVELS+tid - offset];
        else
            temp[pout*GRAYLEVELS+tid] = temp[pin*GRAYLEVELS+tid];
        __syncthreads();
    }

    cdf[tid] = temp[pout*GRAYLEVELS+tid];
}

__device__ unsigned int findMin(unsigned int* cdf){
    unsigned int min = 0;
    for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
    }
    
    return min;
}

__device__ unsigned char Scale(unsigned int cdf, unsigned int cdfmin, unsigned int imageSize){
    float scale;
    scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
    scale = round(scale * (float)(GRAYLEVELS-1));
    return (int)scale;
}

__global__ void equalize_kernel(unsigned char* image_in, unsigned char* image_out, int width, int height, unsigned int* cdf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
        image_out[y * width + x] = Scale(cdf[image_in[y * width + x]], findMin(cdf), width * height);
}

float run(unsigned char* imageIn, int width, int height, int cpp) {
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    //Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *imageOut = (unsigned char *)malloc(height * width * sizeof(unsigned int));
    unsigned int *histogram = (unsigned int *)malloc(GRAYLEVELS * sizeof(unsigned int));
    unsigned int *CDF = (unsigned int *)malloc(GRAYLEVELS * sizeof(unsigned int));

    unsigned char *imageIn_cuda;
    unsigned char *imageOut_cuda;
    unsigned int *histogram_cuda;
    unsigned int *CDF_cuda;
    cudaMalloc((void **)&imageIn_cuda, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&imageOut_cuda, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&histogram_cuda, GRAYLEVELS * sizeof(unsigned int));
    cudaMalloc((void **)&CDF_cuda, GRAYLEVELS * sizeof(unsigned int));

    cudaMemcpy(imageIn_cuda, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 1. Izračun histograma:
    dim3 blockDim(16,16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    histogram_kernel<<<gridDim, blockDim>>>(imageIn_cuda, width, height, histogram_cuda);
    cudaDeviceSynchronize();
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA histogram error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    CDF_kernel<<<1, GRAYLEVELS, GRAYLEVELS * sizeof(unsigned int)>>>(histogram_cuda, CDF_cuda);
    cudaDeviceSynchronize();
    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA CDF error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    equalize_kernel<<<gridDim, blockDim>>>(imageIn_cuda, imageOut_cuda, width, height, CDF_cuda);
    cudaDeviceSynchronize();
    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA equalize error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    cudaMemcpy(imageOut, imageOut_cuda, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    stbi_write_jpg("images-output/output.jpg", width, height, DESIRED_NCHANNELS, imageOut, 100);

    free(imageOut);
    free(histogram);
    free(CDF);
    cudaFree(imageIn_cuda);
    cudaFree(imageOut_cuda);
    cudaFree(histogram_cuda);
    cudaFree(CDF_cuda);

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
}
