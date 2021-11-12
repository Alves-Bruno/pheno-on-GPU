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

int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    int error_code = 1;
    struct stat s;

    if( stat(sInputPath.c_str(), &s) == 0 )
    {
        if( s.st_mode & S_IFREG )
        {
            filelist.push_back(sInputPath);
        }
        else if( s.st_mode & S_IFDIR )
        {
            // processing each file in directory
            DIR *dir_handle;
            struct dirent *dir;
            dir_handle = opendir(sInputPath.c_str());
            std::vector<std::string> filenames;
            if (dir_handle)
            {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL)
                {
                    if (dir->d_type == DT_REG)
                    {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    }
                    else if (dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..")
                        {
                            readInput(sInputPath + sname + "/", filelist);
                        }
                    }
                }
                closedir(dir_handle);
            }
            else
            {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        }
        else
        {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    }
    else
    {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }

    return 0;
}

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

class ImageDecoder {
  public:          
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

  ImageDecoder(std::vector<std::string> images_path, int batch_size, int total_images, int n_streams){

    this->images_path = images_path;
    this->batch_size = batch_size;
    this->total_images = total_images;
    this->n_streams = n_streams;
  }

  void nvJPEG_start(){

     
    
    /*this->status = nvjpegCreateEx(
		      NVJPEG_BACKEND_GPU_HYBRID,
		      &this->dev_allocator,
		      &this->pinned_allocator,
		      NVJPEG_FLAGS_DEFAULT,
		      &this->handle);

    nvjpegJpegStateCreate(this->handle, &this->jpeg_handle);

    nvjpegDecodeBatchedInitialize(
        this->handle, this->jpeg_handle,
	this->batch_size, 1, NVJPEG_OUTPUT_RGB);*/

    nvjpegCreateSimple(&this->handle);

    nvjpegDecoderCreate(
        this->handle, 
	NVJPEG_BACKEND_GPU_HYBRID, 
        &this->decoder_handle);

   nvjpegDecoderStateCreate(
        this->handle,
        this->decoder_handle,
	&this->decoder_state);

   nvjpegJpegStreamCreate(
        this->handle, 
	&this->jpeg_stream);	

   nvjpegBufferPinnedCreate(
        this->handle, 
	&this->pinned_allocator,
	&this->pinned_buffer);

   nvjpegBufferDeviceCreate(
        this->handle, 
        &this->device_allocator,
	&this->device_buffer);

   nvjpegStateAttachPinnedBuffer(
        this->decoder_state,
        this->pinned_buffer);

   nvjpegStateAttachDeviceBuffer(
        this->decoder_state,
        this->device_buffer);

   nvjpegDecodeParamsCreate(
        this->handle, 
        &this->decode_params);

   nvjpegDecodeParamsSetOutputFormat(
        this->decode_params,
        NVJPEG_OUTPUT_RGB);	

  }

  void load_next_images(int buffer_size, int image_index, unsigned char **buffer_images, size_t *buffer_images_sizes){
    nvtxRangePush(__FUNCTION__);

    //#pragma omp parallel default(shared)
{
  //#pragma omp for schedule(static)
    for(int i = 0; i < buffer_size; i++){
      //std::cout << i << " " << input_images_names[i] << std::endl;
      FILE* file = fopen(this->images_path[i+image_index].c_str(), "rb");
      fseek(file, 0, SEEK_END);
      unsigned long size=ftell(file);

      unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
      malloc_check((void*) jpg, "jpg");
      fseek(file, 0, 0);
      fread(jpg, size, 1, file);

      fclose(file);

      buffer_images[i] = jpg;
      buffer_images_sizes[i] = (size_t) size;
    }
 }
    nvtxRangePop();

  }

  void create_output_structs(int buffer_size, int image_index,
			     unsigned char **buffer_images, size_t *buffer_images_sizes,
			     int *buffer_considered_pixels, nvjpegImage_t *buffer_destinations){

    nvtxRangePush(__FUNCTION__);
    for(int i = 0; i < buffer_size; i++){

      int nComponents, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
      nvjpegChromaSubsampling_t subsampling;
      nvjpegGetImageInfo(
			 this->handle,
			 buffer_images[i],
			 buffer_images_sizes[i],
			 &nComponents, &subsampling, widths, heights);
      
      nvjpegImage_t img_info;
      for(int c=0; c<3; c++){
	img_info.pitch[c] = widths[0];

	auto cmalloc_state = cudaMalloc((void**)&img_info.channel[c], widths[0]*heights[0]);
	if(cmalloc_state != cudaSuccess){
	  
	  std::cout << "CudaMalloc error(" << cudaGetErrorString(cmalloc_state) << ") on " << i+image_index << " at "<< this->images_path[i+image_index] << std::endl;
	  exit(1);
	}
      }
      buffer_considered_pixels[i] = widths[0]*heights[0];
      buffer_destinations[i] = img_info;
    }
    
    nvtxRangePop();

  }

  void create_streams(){
    // Create the streams
    this->streams = (cudaStream_t*) malloc(this->n_streams * sizeof(cudaStream_t));
    for(int i=0; i< this->n_streams; i++){
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

  }

  void delete_streams(){
    for(int i = 0; i < this->n_streams; i++){
      cudaStreamDestroy(this->streams[i]);
    }
  }


  void decode_decoupled(){

    int *channel_avg = (int *) malloc(this->total_images * 3 * sizeof(int));
    
    this->create_streams();    
    nvjpegImage_t *device_output_buffer = (nvjpegImage_t*) malloc(sizeof(nvjpegImage_t) * this->batch_size);
    
    for(int b_start = 0; b_start < this->total_images; b_start += this->batch_size){

      nvtxRangePush("batch_decode");

      // Load the next batch of images
      unsigned char **buffer_images = (unsigned char**) malloc(this->batch_size * sizeof(unsigned char *));
      size_t *buffer_images_sizes = (size_t*) malloc(this->batch_size * sizeof(size_t));
      malloc_check((void*) buffer_images, "buffer_images");
      malloc_check((void*)buffer_images_sizes, "buffer_images_sizes");
      this->load_next_images(this->batch_size, b_start, buffer_images, buffer_images_sizes);

      
      nvjpegJpegStream_t *jpeg_streams = (nvjpegJpegStream_t*) malloc(sizeof(nvjpegJpegStream_t) * this->batch_size);
      for(int in_batch = 0; in_batch < this->batch_size; in_batch++){

	nvjpegJpegStreamCreate(
        this->handle, 
	&jpeg_streams[in_batch]);

	// Parses the bitstream and stores the metadata in the jpeg_stream struct. 
	nvjpegJpegStreamParse(
			      this->handle,
			      buffer_images[in_batch], 
			      buffer_images_sizes[in_batch],
			      0,
			      0,
			      jpeg_streams[in_batch]);
	//std::cout << "buffer_image_size: " << buffer_images_sizes[in_batch] << " bytes." << std::endl;
      }

      // Create the device output buffers assuming that all images have the same size
      if(b_start == 0){
	
	for(int in_batch = 0; in_batch < this->batch_size; in_batch++){ 
	  unsigned int components_num;
	  nvjpegJpegStreamGetComponentsNum(
					   jpeg_streams[0],
					   &components_num);
	  //std::cout << "components_num: " << components_num << std::endl;

	  int nComponents, Awidths[components_num], Aheights[components_num];
	  nvjpegChromaSubsampling_t subsampling;
	  
	  nvjpegGetImageInfo(
			 this->handle,
			 buffer_images[in_batch],
			 buffer_images_sizes[in_batch],
			 &nComponents, &subsampling, Awidths, Aheights);
	  
	  //std::cout << "widths[0]: " << Awidths[0] << std::endl;
	  //std::cout << "heights[0]: " << Aheights[0] << std::endl;
      
	  for(int channel = 0; channel < components_num; channel++){
	    unsigned int width;
	    unsigned int height;
	    nvjpegJpegStreamGetComponentDimensions(
						   jpeg_streams[0],
						   0,
						   &width,
						   &height
						   );
	    //std::cout << "width: " << width << std::endl;
	    //std::cout << "heigth: " << height << std::endl;

	    this->images_width = width;
	    this->images_height = height;

	    device_output_buffer[in_batch].pitch[channel] = width;
	    auto malloc_state = cudaMalloc((void**) &(device_output_buffer[in_batch].channel[channel]), sizeof(unsigned char) * width*height);
	    if(malloc_state != cudaSuccess){
	      std::cout << "CudaMalloc error(" << cudaGetErrorString(malloc_state) <<  std::endl;
	      exit(1);
	    }
	  }
	
	}
      }


#pragma omp parallel default(shared)
{
#pragma omp for schedule(static)
      for(int in_batch = 0; in_batch < this->batch_size; in_batch++){

	nvtxRangePush("host_decode");
	// It is synchronous with respect of the host
	nvjpegDecodeJpegHost(
			     this->handle,
			     this->decoder_handle,
			     this->decoder_state,
			     this->decode_params,
			     jpeg_streams[in_batch]);
	nvtxRangePop();
      }
}

// #pragma omp parallel default(shared)
{
  //#pragma omp for schedule(static)
       for(int in_batch = 0; in_batch < this->batch_size; in_batch++){
	 nvtxRangePush("transfer");
	 // Contains both host and device operations.
	 // Hence it is a mix of synchronous and asynchronous operations with respect to the host
	 nvjpegDecodeJpegTransferToDevice(
					  this->handle,
					  this->decoder_handle,
					  this->decoder_state,
					  jpeg_streams[in_batch],
					  this->streams[in_batch % this->n_streams]);
	 nvtxRangePop();
	
       }
}      
      nvtxRangePush("GPU_decode");
      for(int in_batch = 0; in_batch < this->batch_size; in_batch++){
	// This phase is asynchronous with respect to the host
	nvjpegDecodeJpegDevice(
			       this->handle, 
			       this->decoder_handle,
			       this->decoder_state,
			       &device_output_buffer[in_batch],
			       this->streams[in_batch % this->n_streams]);
	
      }
      nvtxRangePop();
      nvtxRangePop();

      /*
	cudaDeviceSynchronize();
      // for end
      for(int in_batch = 0; in_batch < this->batch_size; in_batch++){
	for(int c=0; c<3 ; c++){

	  unsigned char* values = (unsigned char *) malloc(sizeof(unsigned char) * (this->images_width*this->images_height));
	  cudaMemcpy((void*) values, (const void*) device_output_buffer[in_batch].channel[c],
		     (this->images_width*this->images_height), cudaMemcpyDeviceToHost);

	  cudaDeviceSynchronize();
	  for(int bruno=0; bruno<(this->images_width*this->images_height); bruno++){
	      std::cout << (int)bruno << ":";
	      std::cout << (int)values[bruno] << std::endl;

	      } std::cout << std::endl;
	  
	  //std::cout << (this->images_width*this->images_height) << std::endl;
	  thrust::device_ptr<unsigned char> channel((unsigned char*) device_output_buffer[in_batch].channel[c]);
	  
	  //	  channel_avg[(in_batch*3) + c] = thrust::reduce(
	  int channel_sum = thrust::reduce(
			      channel, // Vector start
			      channel + (this->images_width*this->images_height), // Vector end 
			      (int) 0, // reduce first value
			      thrust::plus<int>()); // reduce operation

	  cudaDeviceSynchronize();
	  std::cout << "SUM: " << channel_sum << std::endl;
	  std::cout << "AVG: " << channel_sum / (float) (this->images_width*this->images_height) << std::endl;
	  
	  //	  std::cout << "SUM: " << channel_avg[(in_batch*3) + c] << std::endl;
	  //	  std::cout << "AVG: " << channel_avg[(in_batch*3) + c] / (float) (this->images_width*this->images_height) << std::endl;
	}
      }*/

    }

    //    cudaFree();
  }

  void decode(){

    this->create_streams();
    
    int stream_to_use = 0;

#pragma omp parallel default(shared)
{
#pragma omp for schedule(dynamic)
    for(int b_start = 0; b_start < this->total_images; b_start += this->batch_size){
      //int tid = omp_get_thread_num();
      //std::cout << tid << ": " << b_start << std::endl;
      nvtxRangePush(__FUNCTION__);

      // Set the current batch size of images
      int b_size = 0;
      if(b_start + this->batch_size >= this->total_images){
	b_size = this->total_images - b_start;
	nvjpegDecodeBatchedInitialize(
            this->handle, this->jpeg_handle,
	    b_size, 1, NVJPEG_OUTPUT_RGB);
      }else{
	b_size = this->batch_size;
      }

#pragma omp critical
{
      if(stream_to_use >= this->n_streams){
	stream_to_use = 0;
      }
}
      // Load the next batch of images
      unsigned char **buffer_images = (unsigned char**) malloc(b_size * sizeof(unsigned char *));
      size_t *buffer_images_sizes = (size_t*) malloc(b_size * sizeof(size_t));
      malloc_check((void*) buffer_images, "buffer_images");
      malloc_check((void*)buffer_images_sizes, "buffer_images_sizes");
      this->load_next_images(b_size, b_start, buffer_images, buffer_images_sizes);

#pragma omp critical
{
  // Create the output structs for the next decode
      int *buffer_considered_pixels = (int*) malloc(sizeof(int) * b_size);
      nvjpegImage_t *buffer_destinations = (nvjpegImage_t*) malloc(b_size * sizeof(nvjpegImage_t));
      malloc_check((void*) buffer_destinations, "buffer_destinations");
      this->create_output_structs(b_size, b_start,
			    buffer_images, buffer_images_sizes,
			    buffer_considered_pixels, buffer_destinations);
      
      nvjpegDecodeBatched(
			  this->handle,
			  this->jpeg_handle,
			  buffer_images,
			  buffer_images_sizes,
			  buffer_destinations,
			  this->streams[stream_to_use]);
      //std::cout << "Use this stream: " << stream_to_use << std::endl;
      stream_to_use++;
      

      size_t free_mem;
      size_t total_mem;
      cudaMemGetInfo(&free_mem , &total_mem);
      //std::cout << "Free: " << free_mem << " bytes of " << total_mem << std::endl;
      //std::cout << "Free memory: " << 100.00 * (free_mem / float(total_mem)) << "%" << std::endl;

      nvtxMark("cudaFree");
      for(int i = 0; i < b_size; i++){
	free(buffer_images[i]);
	for(int channel=0; channel < 3; channel++){
	  cudaFree(buffer_destinations[i].channel[channel]);
	}
      }
      free(buffer_images);
      free(buffer_images_sizes);
      free(buffer_destinations);

      nvtxRangePop();

    }
}
}
    this->delete_streams();
  }
  
};

int main(int argc, char *argv[]){

  if(argc < 5){
    std::cout << "Usage: " << argv[0] <<
      " <images_dir> <batch_size> <number_of_images> <number_of_streams>" << std::endl;
    exit(1);
  }
  std::vector<std::string> input_images_names;
  const std::string images_path = argv[1];

  int batch_size = atoi(argv[2]);
  int total_images = atoi(argv[3]);
  int n_streams = atoi(argv[4]);
  
  // Read images at path
  readInput(images_path, input_images_names);

  if(total_images > input_images_names.size())
    total_images = input_images_names.size();

  if(batch_size > total_images)
    batch_size = total_images;

  ImageDecoder decoder(input_images_names, batch_size, total_images, n_streams);
  decoder.nvJPEG_start();
  decoder.decode_decoupled();

  return(0);

}
