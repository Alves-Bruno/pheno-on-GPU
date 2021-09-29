#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <string.h>  // strcmpi
#ifndef _WIN64
#include <sys/time.h>  // timings
#include <unistd.h>
#endif
#include <dirent.h>  
#include <sys/stat.h>
#include <cuda.h>
#include <sys/types.h>


#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }


typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char> > FileData;

struct decode_params_t {
  std::string input_dir;
  int batch_size;
  int total_images;
  int dev;
  int warmup;

  nvjpegJpegState_t nvjpeg_state;
  nvjpegHandle_t nvjpeg_handle;
  cudaStream_t stream;

  // used with decoupled API
  nvjpegJpegState_t nvjpeg_decoupled_state;
  nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
  nvjpegBufferDevice_t device_buffer;
  nvjpegJpegStream_t  jpeg_streams[2]; //  2 streams for pipelining
  nvjpegDecodeParams_t nvjpeg_decode_params;
  nvjpegJpegDecoder_t nvjpeg_decoder;

  nvjpegOutputFormat_t fmt;
  bool write_decoded;
  std::string output_dir;

  bool hw_decode_available;
};

int dev_malloc(void **p, size_t s);
int dev_free(void *p);
int host_malloc(void** p, size_t s, unsigned int f);
int host_free(void* p);

int read_next_batch(FileNames &image_names, int batch_size,
                    FileNames::iterator &cur_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &current_names);
  
// prepare buffers for RGBi output format
int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len,
                    std::vector<int> &img_width, std::vector<int> &img_height,
                    std::vector<nvjpegImage_t> &ibuf,
                    std::vector<nvjpegImage_t> &isz, FileNames &current_names,
                    decode_params_t &params);

void create_decoupled_api_handles(decode_params_t& params);

void destroy_decoupled_api_handles(decode_params_t& params);

void release_buffers(std::vector<nvjpegImage_t> &ibuf);

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params,
                  double &time);

double process_images(FileNames &image_names, decode_params_t &params,
                      double &total);
