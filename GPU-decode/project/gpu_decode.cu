#include "gpu_decode.h"

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

ImageDecoder::ImageDecoder(
			    std::vector<std::string> images_path,
			    int batch_size, int total_images, int n_streams){

  nvtxRangePush(__FUNCTION__);
    this->images_path = images_path;
    this->batch_size = batch_size;
    this->total_images = total_images;
    this->n_streams = n_streams;
    nvtxRangePop();
}

void ImageDecoder::nvjpeg_start(){

  nvtxRangePush(__FUNCTION__);
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

  nvtxRangePop();
}

void ImageDecoder::create_gpu_streams(){
  nvtxRangePush(__FUNCTION__);
  // Create the streams
  this->streams = (cudaStream_t*) malloc(this->n_streams * sizeof(cudaStream_t));
  for(int i=0; i< this->n_streams; i++){
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  nvtxRangePop();
}

void ImageDecoder::delete_gpu_streams(){
  nvtxRangePush(__FUNCTION__);
  for(int i = 0; i < this->n_streams; i++){
    cudaStreamDestroy(this->streams[i]);
  }
  nvtxRangePop();
}

void ImageDecoder::load_batch_to_host(
				      image_file *images_buffer,
				      int start,
				      int n_images)
{
  nvtxRangePush(__FUNCTION__);

  //#pragma omp parallel default(shared)
  //{
  //#pragma omp for schedule(static)
  for(int i = 0; i < n_images; i++){

    FILE* file = fopen(this->images_path[start + i].c_str(), "rb");
    fseek(file, 0, SEEK_END);
    unsigned long size=ftell(file);

    unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
    malloc_check((void*) jpg, "jpg");
    fseek(file, 0, 0);
    fread(jpg, size, 1, file);

    fclose(file);

    //    image_file image;
    //    image.bitstream = jpg;
    //    image.size = (size_t) size;

    //#pragma omp critical
    //{
  images_buffer[i].bitstream = jpg;
  images_buffer[i].size = size;
  //}
 
 }
  //}
  nvtxRangePop();
}

nvjpegJpegStream_t *ImageDecoder::create_JPEG_streams(
						      image_file *images_buffer,
						      unsigned int images_buffer_size
){

  nvtxRangePush(__FUNCTION__);
  nvjpegJpegStream_t *jpeg_streams = (nvjpegJpegStream_t*)
    malloc(sizeof(nvjpegJpegStream_t) * images_buffer_size);
  
  for(int i = 0; i < images_buffer_size; i++){

    nvjpegJpegStreamCreate(
			   this->handle, 
			   &jpeg_streams[i]);

    // Parses the bitstream and stores the metadata in the jpeg_stream struct. 
    nvjpegJpegStreamParse(
			  this->handle,
			  images_buffer[i].bitstream, 
			  images_buffer[i].size,
			  0,
			  0,
			  jpeg_streams[i]);
    //std::cout << "buffer_image_size: " << buffer_images_sizes[in_batch] << " bytes." << std::endl;
  }
  nvtxRangePop();
  return(jpeg_streams);
  
}

nvjpegImage_t *ImageDecoder::malloc_output_buffer(
				   unsigned int channels,
				   unsigned int width,
				   unsigned int height,
				   unsigned int n_images
){

  nvtxRangePush(__FUNCTION__);
  nvjpegImage_t *output_buffer = (nvjpegImage_t*)
                malloc(sizeof(nvjpegImage_t) * n_images);

  // Check GPU memory
  unsigned int image_size_dev_mem = channels * width * height;
  size_t dev_mem_free, dev_mem_total;
  cudaMemGetInfo(&dev_mem_free, &dev_mem_total);
  // Free mem = dev_free_mem - ((thrust_usage) + processing buffers)
  dev_mem_free = dev_mem_free - (image_size_dev_mem + (0.2 * dev_mem_free));
  int max_batch_size = dev_mem_free / (image_size_dev_mem * 2);

  if(n_images > max_batch_size){
    this->batch_size = max_batch_size;
    n_images = max_batch_size;
    std::cout << "WARNING: batch_size greater than device memory capacity." << std::endl;
    std::cout << "Altering batch size to: " << max_batch_size << std::endl;
  }
  
  for(int i = 0; i < n_images; i++){
    for(int c = 0; c < channels; c++){
      
      output_buffer[i].pitch[c] = width;
      auto malloc_state = cudaMalloc((void**) &(output_buffer[i].channel[c]),
				     sizeof(unsigned char) * width*height);
      
      if(malloc_state != cudaSuccess){
	std::cout << "CudaMalloc error(" << cudaGetErrorString(malloc_state) <<  std::endl;
	exit(1);
      }

    }
  }
  nvtxRangePop();
  return(output_buffer);
}

nvjpegImage_t *ImageDecoder::prepare_decode(){
  
  nvtxRangePush(__FUNCTION__);
  this->nvjpeg_start();
  this->create_gpu_streams();

  // Load first image
  image_file *first_image_buffer = (image_file*) malloc(sizeof(image_file) * 1);
  this->load_batch_to_host(first_image_buffer, 0, 1);
  nvjpegJpegStream_t *first_image_stream;
  first_image_stream = create_JPEG_streams(first_image_buffer, 1);
  nvjpegImage_t *output_buffer;

  unsigned int width;
  unsigned int height;
  unsigned int nchannels = 3;
  // Get dimensions of the first image
  // Assuming all the images have the same dimensions
  nvjpegJpegStreamGetComponentDimensions(
					 first_image_stream[0],
					 0,
					 &width,
					 &height
					 );

  this->images_width = width;
  this->images_height = height;
  this->considered_pixels = width * height;

  output_buffer = malloc_output_buffer(nchannels, width, height, this->batch_size);

  free(first_image_buffer[0].bitstream);
  free(first_image_stream);
  free(first_image_buffer);
  
  nvtxRangePop();
  return(output_buffer);
    
}

void ImageDecoder::decode_batch(
			   nvjpegJpegStream_t *JPEG_stream_buffer,
			   nvjpegImage_t *output_buffer,
			   int n_images			
){

  nvtxRangePush(__FUNCTION__);

  //#pragma omp parallel default(shared)
  //  {
    //#pragma omp for schedule(static)  
  for(int i = 0; i < n_images; i++){

    nvtxRangePush("HOST_decode");
    // It is synchronous with respect of the host
    nvjpegDecodeJpegHost(
			 this->handle,
			 this->decoder_handle,
			 this->decoder_state,
			 this->decode_params,
			 JPEG_stream_buffer[i]);
    nvtxRangePop();
  }
  //}
 
  for(int i = 0; i < n_images; i++){
    nvtxRangePush("transfer");
    // Contains both host and device operations.
    // Hence it is a mix of synchronous and asynchronous operations with respect to the host
    nvjpegDecodeJpegTransferToDevice(
				     this->handle,
				     this->decoder_handle,
				     this->decoder_state,
				     JPEG_stream_buffer[i],
				     this->streams[i % this->n_streams]);
    nvtxRangePop();

    // This phase is asynchronous with respect to the host
    nvtxRangePush("GPU_decode");
    nvjpegDecodeJpegDevice(
			   this->handle, 
			   this->decoder_handle,
			   this->decoder_state,
			   &output_buffer[i],
			   this->streams[i % this->n_streams]);
    nvtxRangePop();

  }
  nvtxRangePop();
}

void ImageDecoder::batch_avg(float *batch_averages,
			     nvjpegImage_t *output_buffer, int channels, int start, int n_images){

  nvtxRangePush(__FUNCTION__);

  //  std::vector<float> batch_averages;
  
  for(int i = 0; i < n_images; i++){
    for(int c = 0; c < channels; c++){
      thrust::device_ptr<unsigned char> channel((unsigned char*)
						output_buffer[i].channel[c]);
      
      int channel_sum = thrust::reduce(
		          channel, // Vector start
			  channel + (this->considered_pixels), // Vector end 
			  (int) 0, // reduce first value
			  thrust::plus<int>()); // reduce operation

      batch_averages[((start + i)*channels) + c] = (channel_sum / (float) this->considered_pixels);
    }
  }
  nvtxRangePop();
}

void ImageDecoder::decode_and_calculate(float* all_avg_values, bool show_rgb){

  if(show_rgb)
    printf("image_id, x, y, R, G, B\n");

  //  std::vector<float> all_avg_values;
  nvjpegImage_t *output_buffer = this->prepare_decode();

  //float* batch_avg = malloc(sizeof(float) * this->total_images);

  // For each batch
  for(int i = 0; i < this->total_images; i += this->batch_size){

    // Set current batch size
    int bsize = this->batch_size;
    if(i + this->batch_size >= this->total_images)
      bsize = this->total_images - i;

    image_file *images_buffer = (image_file*) malloc(sizeof(image_file) * bsize);
    this->load_batch_to_host(images_buffer, i, bsize);
    nvjpegJpegStream_t *JPEG_streams = create_JPEG_streams(images_buffer, (unsigned int)bsize);
    
    this->decode_batch(JPEG_streams, output_buffer, bsize);
    cudaDeviceSynchronize();

    this->batch_avg(all_avg_values, output_buffer, 3, i, bsize);
    //all_avg_values.insert(all_avg_values.end(), batch_avg.begin(), batch_avg.end());
    cudaDeviceSynchronize();

    if(show_rgb){
      for(int in_batch = 0; in_batch < this->batch_size; in_batch++){

      	
	std::vector<unsigned char*> canais_rgb;
	
	for(int c=0; c<3 ; c++){
	  unsigned char* values = (unsigned char *) malloc(sizeof(unsigned char)
							   * (this->considered_pixels));
	  cudaMemcpy((void*) values, (const void*) output_buffer[in_batch].channel[c],
		     (this->considered_pixels), cudaMemcpyDeviceToHost);
	
	  cudaDeviceSynchronize();
	  canais_rgb.push_back(values);
	}

	for(int bruno=0; bruno<(this->considered_pixels); bruno++){
	  printf("%d, %d, %d, %d, %d, %d\n", in_batch, bruno / this->images_height,
		 bruno % this->images_width, canais_rgb[0][bruno],
		 canais_rgb[1][bruno], canais_rgb[2][bruno]);	  
	}

      }
    }

    // Free host memory stuff
    for(int j = 0; j < bsize; j++){
      free(images_buffer[j].bitstream);
    }
    free(JPEG_streams);
    free(images_buffer);
    
  }
  nvtxRangePop();
}
