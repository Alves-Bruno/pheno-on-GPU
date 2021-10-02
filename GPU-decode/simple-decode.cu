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

#include <chrono> // To record times

__global__ void my_kernel(unsigned char* copy, unsigned char* red, unsigned char* green, unsigned char* blue){
  copy[0] = red[0];
  copy[1] = green[0];
  copy[2] = blue[0];

  printf("cor: %d-%d-%d\n", (int)red[0], (int)green[0], (int)blue[0]);
}

int main(int argc, char *argv[]){

  std::chrono::_V2::system_clock::time_point start_fread = std::chrono::high_resolution_clock::now();

  FILE* file = fopen(argv[1], "rb");
  fseek(file, 0, SEEK_END);
  unsigned long size=ftell(file);

  unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
  fseek(file, 0, 0);
  fread(jpg, size, 1, file);

  std::chrono::_V2::system_clock::time_point end_fread = std::chrono::high_resolution_clock::now();

  std::chrono::_V2::system_clock::time_point start_decode = std::chrono::high_resolution_clock::now();
  nvjpegHandle_t handle;
  nvjpegCreateSimple(&handle);
  nvjpegJpegState_t jpeg_handle;
  nvjpegJpegStateCreate(handle, &jpeg_handle);

  int nComponents, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
  nvjpegChromaSubsampling_t   subsampling;
  nvjpegGetImageInfo(handle, jpg, size, &nComponents, &subsampling, widths, heights);
  //  for(int i=0; i<nComponents; i++){
  //  printf("%d: %d %d\n", i, widths[i], heights[i]);
  //}
  nvjpegImage_t img_info;
  for(int i=0; i<3; i++){
    img_info.pitch[i] = widths[0];
    cudaMalloc((void**)&img_info.channel[i], widths[0]*heights[0]);
  }
  nvjpegDecode(handle, jpeg_handle, jpg, size,
  	NVJPEG_OUTPUT_RGB,
  	&img_info,
  	0);
  //printf("%p %p %p\n", &img_info, img_info.pitch, img_info.channel[0]);

  cudaDeviceSynchronize();
  std::chrono::_V2::system_clock::time_point end_decode = std::chrono::high_resolution_clock::now();

  /*  unsigned char *red_char_host = (unsigned char*) malloc(sizeof(unsigned char) * (widths[0]*heights[0]));
  cudaError_t copy_err = cudaMemcpy((void*)red_char_host, (void*)img_info.channel[0], (widths[0]*heights[0]) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
  
  if (copy_err == cudaSuccess){
    printf("COPY SUCCESS\n");
    int sum_check = 0;
    for(int i=0; i<(widths[0]*heights[0]); i++){
      //printf("[%d]: %d\n", i, red_char_host[i]);
      sum_check += (int) red_char_host[i];
    }
    printf("RED SUM CHECK: %d\n", sum_check);
    }*/

  std::chrono::_V2::system_clock::time_point start_calc = std::chrono::high_resolution_clock::now();
  int channel_sum[3];
  for(int i=0; i<3; i++){
    
    thrust::device_ptr<unsigned char> channel((unsigned char*)img_info.channel[i]);
    // compute sum on the device
    channel_sum[i] = thrust::reduce(
      channel, // Vector start
      channel + (widths[0]*heights[0]), // Vector end 
      (int) 0, // reduce first value
      thrust::plus<int>()); // reduce operation
    
  }
  cudaDeviceSynchronize();
  std::chrono::_V2::system_clock::time_point end_calc = std::chrono::high_resolution_clock::now();
  
  //  for(int i=0; i<3; i++){
  //  printf("Channel[%d] sum: %d\n", i, channel_sum[i]);
  //  printf("Channel[%d] avg: %f\n", i, channel_sum[i] / (float)(widths[0]*heights[0]));
  //}

  double fread_time = std::chrono::duration_cast<std::chrono::microseconds>(end_fread - start_fread).count();
  double decode_time = std::chrono::duration_cast<std::chrono::microseconds>(end_decode - start_decode).count();
  double calc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_calc - start_calc).count();

  printf("%s, %lf, %lf, %lf\n", argv[1], fread_time, decode_time, calc_time);
}
