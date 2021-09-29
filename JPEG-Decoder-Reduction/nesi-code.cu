#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

__global__ void my_kernel(unsigned char* red, unsigned char* green, unsigned char* blue){
  //printf("cor: %d-%d-%d\n", (int)red[0], (int)green[0], (int)blue[0]);
}

__global__ unsigned char my_kernel_c(unsigned char* channel){
  //printf("cor: %d-%d-%d\n", (int)red[0], (int)green[0], (int)blue[0]);
  return(channel[0]);
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
    printf("%d %d\n", widths[i], heights[i]);
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
  unsigned char pixel = my_kernel_c<<<1,1>>>(img_info.channel[0]);

  cudaDeviceSynchronize();
  printf("Pixel: %c\n", pixel);

}
