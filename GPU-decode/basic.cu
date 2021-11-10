#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <vector>

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

#include <dirent.h>  
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>

#include <nvToolsExt.h> 
#include <sys/syscall.h>
#include <unistd.h>

#include <omp.h>


int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }
int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
int host_free(void* p) { return (int)cudaFreeHost(p); }

void malloc_check(void *ptr, const char *message){
  if(ptr == NULL){
    std::cout << "Could not malloc [" << ptr << "]" << std::endl;
    exit(1);
  }
}

int main(int argc, char *argv[]){

  // nvJPEG stuff
  nvjpegHandle_t handle;
  nvjpegDevAllocator_t device_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};
  //nvjpegStatus_t status;
  //nvjpegJpegState_t jpeg_handle;
  nvjpegJpegDecoder_t decoder_handle;
  nvjpegJpegState_t decoder_state;
  nvjpegJpegStream_t jpeg_stream;
  //nvjpegPinnedAllocator_t pinned_allocator;
  nvjpegBufferPinned_t pinned_buffer;
  //nvjpegDevAllocator_t device_allocator;
  nvjpegBufferDevice_t device_buffer;
  nvjpegDecodeParams_t decode_params;
  
  //cudaStream_t *streams;

  // Create
  nvjpegCreateSimple(&handle);
  nvjpegDecoderCreate(
		      handle, 
		      NVJPEG_BACKEND_GPU_HYBRID, 
		      &decoder_handle);
  nvjpegDecoderStateCreate(
			   handle,
			   decoder_handle,
			   &decoder_state);
  //  nvjpegJpegStreamCreate(
  //			 handle, 
  //			 &jpeg_stream);	
  nvjpegBufferPinnedCreate(
			   handle, 
			   &pinned_allocator,
			   &pinned_buffer);
  nvjpegBufferDeviceCreate(
			   handle, 
			   &device_allocator,
			   &device_buffer);

  nvjpegStateAttachPinnedBuffer(
				decoder_state,
				pinned_buffer);

  nvjpegStateAttachDeviceBuffer(
				decoder_state,
				device_buffer);

  nvjpegDecodeParamsCreate(
			   handle, 
			   &decode_params);

  nvjpegDecodeParamsSetOutputFormat(
				    decode_params,
				    NVJPEG_OUTPUT_RGB);	

  FILE* file = fopen("/home/users/bsalves/pheno-on-GPU/GPU-decode/input/image.jpg", "rb");
  fseek(file, 0, SEEK_END);
  unsigned long size=ftell(file);

  unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
  malloc_check((void*) jpg, "jpg");
  fseek(file, 0, 0);
  fread(jpg, size, 1, file);
  fclose(file);

  //nvjpegJpegStream_t jpeg_stream;

  nvjpegJpegStreamCreate(
			 handle, 
			 &jpeg_stream);

  nvjpegJpegStreamParse(
			handle,
			jpg, 
		        size,
			0,
			1,
			jpeg_stream);

  nvjpegImage_t output;

  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  nvjpegGetImageInfo(
        handle, jpg, size,
        &channels, &subsampling, widths, heights);

  for (int c = 0; c < channels; c++) {
    std::cout << "Channel #" << c << " size: " << widths[c] << " x "
	      << heights[c] << std::endl;
  } 

  switch (subsampling) {
  case NVJPEG_CSS_444:
    std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
    break;
  case NVJPEG_CSS_440:
    std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
    break;
  case NVJPEG_CSS_422:
        std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
        break;
  case NVJPEG_CSS_420:
    std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
    break;
  case NVJPEG_CSS_411:
        std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
        break;
  case NVJPEG_CSS_410:
        std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
        break;
      case NVJPEG_CSS_GRAY:
        std::cout << "Grayscale JPEG " << std::endl;
        break;
  case NVJPEG_CSS_UNKNOWN:
    std::cout << "Unknown chroma subsampling" << std::endl;
    return EXIT_FAILURE;
  }

  int buffer_width = widths[0];
  int buffer_height = heights[0];
  int buffer_size = buffer_width * buffer_height;
  std::cout << "buffer_size: " << buffer_size << std::endl;
  
  for(int i=0; i<3; i++){
    auto malloc_state = cudaMalloc(&output.channel[i], buffer_size);
    if(malloc_state != cudaSuccess){
      std::cout << "CudaMalloc error(" << cudaGetErrorString(malloc_state) <<  std::endl;
      exit(1);
    }
    }

  cudaDeviceSynchronize();
  nvjpegDecodeJpegHost(
		       handle,
		       decoder_handle,
		       decoder_state,
		       decode_params,
		       jpeg_stream);

  cudaDeviceSynchronize();
  
  nvjpegDecodeJpegTransferToDevice(
				   handle,
				   decoder_handle,
				   decoder_state,
				   jpeg_stream,
				   0);

  cudaDeviceSynchronize();
  nvjpegDecodeJpegDevice(
			 handle, 
			 decoder_handle,
			 decoder_state,
			 &output,
			 0);
	
  cudaDeviceSynchronize();
  
  // for end
  for(int c=0; c<3 ; c++){

    unsigned char* values = (unsigned char *) malloc(sizeof(unsigned char) * (buffer_size));
    cudaMemcpy((void*) values, (const void*) output.channel[c],
	       buffer_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    for(int bruno=0; bruno<25; bruno++){
      std::cout << bruno << ":";
      std::cout << (int)values[bruno] << std::endl;

    } std::cout << std::endl;
		
  }

  return(0);
}


