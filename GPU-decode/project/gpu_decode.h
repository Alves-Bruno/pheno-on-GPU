#ifndef gpu_decode_h
#define gpu_decode_h

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

#define CHECK_NVJPEG(call)						\
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

int dev_malloc(void **p, size_t s);
int dev_free(void *p);
int host_malloc(void** p, size_t s, unsigned int f);
int host_free(void* p);
void malloc_check(void *ptr, const char *message);


struct image_file {  
    unsigned char *bitstream; 
    size_t size;
};


class ImageDecoder {

  private:          

  unsigned int considered_pixels = 0; 
  
  std::vector<std::string> images_path;
  int batch_i = 0;
  int batch_size;
  int total_images;
  int n_streams;

  int images_width;
  int images_height;

  // nvJPEG stuff
  nvjpegHandle_t handle;
  nvjpegDevAllocator_t device_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};
  nvjpegStatus_t status;
  nvjpegJpegState_t jpeg_handle;
  nvjpegJpegDecoder_t decoder_handle;
  nvjpegJpegState_t decoder_state;
  nvjpegJpegStream_t jpeg_stream;
  //nvjpegPinnedAllocator_t pinned_allocator;
  nvjpegBufferPinned_t pinned_buffer;
  //nvjpegDevAllocator_t device_allocator;
  nvjpegBufferDevice_t device_buffer;
  nvjpegDecodeParams_t decode_params;
  
  cudaStream_t *streams;

 private:

  void nvjpeg_start();
  void create_gpu_streams();
  void delete_gpu_streams();
  void load_batch_to_host(
			  image_file *images_buffer,
			  int start,
			  int n_images);
  nvjpegJpegStream_t *create_JPEG_streams(
					  image_file *images_buffer,
					  unsigned int images_buffer_size);

  nvjpegImage_t *malloc_output_buffer(
				      unsigned int channels,
				      unsigned int width,
				      unsigned int height,
				      unsigned int n_images);

  nvjpegImage_t *prepare_decode();

  void decode_batch(
		    nvjpegJpegStream_t *JPEG_stream_buffer,
		    nvjpegImage_t *output_buffer,
		    int n_images);

  void batch_avg(
		 float* batch_averages,
		 nvjpegImage_t *output_buffer,
		 int channels, int start, int n_images);
  
 public:

  ImageDecoder(std::vector<std::string> images_path,
	       int batch_size, int total_images, int n_streams);

  void decode_and_calculate(float*, bool show_rgb);
  
};


#endif



