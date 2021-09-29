#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>



__global__ void my_kernel(unsigned char* copy, unsigned char* red, unsigned char* green, unsigned char* blue){
  copy[0] = red[0];
  copy[1] = green[0];
  copy[2] = blue[0];

  printf("cor: %d-%d-%d\n", (int)red[0], (int)green[0], (int)blue[0]);
}

int main(){
  nvjpegHandle_t handle;
  nvjpegCreateSimple(&handle);
  nvjpegJpegState_t jpeg_handle;
  nvjpegJpegStateCreate(handle, &jpeg_handle);

  FILE* file = fopen("x.jpg", "rb");
  fseek(file, 0, SEEK_END);
  unsigned long size=ftell(file);

  unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
  fseek(file, 0, 0);
  fread(jpg, size, 1, file);

  int nComponents, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
  nvjpegChromaSubsampling_t   subsampling;
  nvjpegGetImageInfo(handle, jpg, size, &nComponents, &subsampling, widths, heights);
  for(int i=0; i<nComponents; i++){
    printf("%d: %d %d\n", i, widths[i], heights[i]);
  }
  nvjpegImage_t img_info;
  for(int i=0; i<3; i++){
    img_info.pitch[i] = widths[0];
    cudaMalloc((void**)&img_info.channel[i], widths[0]*heights[0]);
  }
  nvjpegDecode(handle, jpeg_handle, jpg, size,
  	NVJPEG_OUTPUT_RGB,
  	&img_info,
  	0);
  printf("%p %p %p\n", &img_info, img_info.pitch, img_info.channel[0]);

  cudaDeviceSynchronize();

  // wrap raw pointer with a device_ptr 
  thrust::device_ptr<unsigned int> red_channel((unsigned int*)img_info.channel[0]);

  // compute sum on the device
  int r_sum = thrust::reduce(
    red_channel, // Vector start
    red_channel + (widths[0]*heights[0]), // Vector end 
    (unsigned int) 0, // reduce first value
    thrust::plus<int>()); // reduce operation

  printf("Red channel sum: %f\n", r_sum / (float)(widths[0]*heights[0]));

  cudaDeviceSynchronize();
  
}
