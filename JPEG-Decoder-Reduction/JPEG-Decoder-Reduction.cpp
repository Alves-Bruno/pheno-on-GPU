#include "JPEG-Decoder-Reduction.h"

int dev_malloc(void **p, size_t s) {
 return (int)cudaMalloc(p, s); 
}

int dev_free(void *p) { 
  return (int)cudaFree(p); 
}

int host_malloc(void** p, size_t s, unsigned int f) {
  return (int)cudaHostAlloc(p, s, f);
}

int host_free(void* p) { 
  return (int)cudaFreeHost(p); 
}

int read_next_batch(FileNames &image_names, int batch_size,
                    FileNames::iterator &cur_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &current_names) {
  int counter = 0;

  while (counter < batch_size) {
    if (cur_iter == image_names.end()) {
      std::cerr << "Image list is too short to fill the batch, adding files "
                   "from the beginning of the image list"
                << std::endl;
      cur_iter = image_names.begin();
    }

    if (image_names.size() == 0) {
      std::cerr << "No valid images left in the input list, exit" << std::endl;
      return EXIT_FAILURE;
    }

    // Read an image from disk.
    std::ifstream input(cur_iter->c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
      std::cerr << "Cannot open image: " << *cur_iter
                << ", removing it from image list" << std::endl;
      image_names.erase(cur_iter);
      continue;
    }

    // Get the size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (raw_data[counter].size() < file_size) {
      raw_data[counter].resize(file_size);
    }
    if (!input.read(raw_data[counter].data(), file_size)) {
      std::cerr << "Cannot read from file: " << *cur_iter
                << ", removing it from image list" << std::endl;
      image_names.erase(cur_iter);
      continue;
    }
    raw_len[counter] = file_size;

    current_names[counter] = *cur_iter;

    counter++;
    cur_iter++;
  }
  return EXIT_SUCCESS;
}

// prepare buffers for RGBi output format
int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len,
                    std::vector<int> &img_width, std::vector<int> &img_height,
                    std::vector<nvjpegImage_t> &ibuf,
                    std::vector<nvjpegImage_t> &isz, FileNames &current_names,
                    decode_params_t &params) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  for (int i = 0; i < file_data.size(); i++) {
    CHECK_NVJPEG(nvjpegGetImageInfo(
        params.nvjpeg_handle, (unsigned char *)file_data[i].data(), file_len[i],
        &channels, &subsampling, widths, heights));

    img_width[i] = widths[0];
    img_height[i] = heights[0];

    std::cout << "Processing: " << current_names[i] << std::endl;
    std::cout << "Image is " << channels << " channels." << std::endl;
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

    int mul = 1;
    // in the case of interleaved RGB output, write only to single channel, but
    // 3 samples at once
    if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI) {
      channels = 1;
      mul = 3;
    }
    // in the case of rgb create 3 buffers with sizes of original image
    else if (params.fmt == NVJPEG_OUTPUT_RGB ||
             params.fmt == NVJPEG_OUTPUT_BGR) {
      channels = 3;
      widths[1] = widths[2] = widths[0];
      heights[1] = heights[2] = heights[0];
    }

    // realloc output buffer if required
    for (int c = 0; c < channels; c++) {
      int aw = mul * widths[c];
      int ah = heights[c];
      int sz = aw * ah;
      ibuf[i].pitch[c] = aw;
      if (sz > isz[i].pitch[c]) {
        if (ibuf[i].channel[c]) {
          CHECK_CUDA(cudaFree(ibuf[i].channel[c]));
        }
        CHECK_CUDA(cudaMalloc(&ibuf[i].channel[c], sz));
        isz[i].pitch[c] = sz;
      }
    }
  }
  return EXIT_SUCCESS;
}

void create_decoupled_api_handles(decode_params_t& params){

  CHECK_NVJPEG(nvjpegDecoderCreate(params.nvjpeg_handle, NVJPEG_BACKEND_DEFAULT, &params.nvjpeg_decoder));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(params.nvjpeg_handle, params.nvjpeg_decoder, &params.nvjpeg_decoupled_state));   
  
  CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL, &params.pinned_buffers[0]));
  CHECK_NVJPEG(nvjpegBufferPinnedCreate(params.nvjpeg_handle, NULL, &params.pinned_buffers[1]));
  CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle, NULL, &params.device_buffer));
  
  CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[0]));
  CHECK_NVJPEG(nvjpegJpegStreamCreate(params.nvjpeg_handle, &params.jpeg_streams[1]));

  CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &params.nvjpeg_decode_params));
}

void destroy_decoupled_api_handles(decode_params_t& params){  

  CHECK_NVJPEG(nvjpegDecodeParamsDestroy(params.nvjpeg_decode_params));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[0]));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_streams[1]));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[0]));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffers[1]));
  CHECK_NVJPEG(nvjpegBufferDeviceDestroy(params.device_buffer));
  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_decoupled_state));  
  CHECK_NVJPEG(nvjpegDecoderDestroy(params.nvjpeg_decoder));
}

void release_buffers(std::vector<nvjpegImage_t> &ibuf) {
  for (int i = 0; i < ibuf.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
      if (ibuf[i].channel[c]) CHECK_CUDA(cudaFree(ibuf[i].channel[c]));
  }
}

// Kernel definition
__global__ void Metric_Calculator(nvjpegImage_t *image)
{
  //int i = threadIdx.x;
  //C[i] = A[i] + B[i];
  printf("From Kernel: %d \n", image->channel[0]);
  
}

__global__ void my_kernel(unsigned char* red, unsigned char* green, unsigned char* blue){
  printf("cor: %d-%d-%d\n", (int)red[0], (int)green[0], (int)blue[0]);
}

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params,
                  double &time) {
  CHECK_CUDA(cudaStreamSynchronize(params.stream));
  cudaEvent_t startEvent = NULL, stopEvent = NULL;
  float loopTime = 0; 
  
  CHECK_CUDA(cudaEventCreate(&startEvent, cudaEventBlockingSync));
  CHECK_CUDA(cudaEventCreate(&stopEvent, cudaEventBlockingSync));


  std::vector<const unsigned char*> batched_bitstreams;
  std::vector<size_t> batched_bitstreams_size;
  std::vector<nvjpegImage_t>  batched_output;

  // bit-streams that batched decode cannot handle
  std::vector<const unsigned char*> otherdecode_bitstreams;
  std::vector<size_t> otherdecode_bitstreams_size;
  std::vector<nvjpegImage_t> otherdecode_output;

  if(params.hw_decode_available){
    for(int i = 0; i < params.batch_size; i++){
      // extract bitstream meta data to figure out whether a bit-stream can be decoded
      nvjpegJpegStreamParseHeader(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i], params.jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0], &isSupported);

      if(isSupported == 0){
        batched_bitstreams.push_back((const unsigned char *)img_data[i].data());
        batched_bitstreams_size.push_back(img_len[i]);
        batched_output.push_back(out[i]);
      } else {
        otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
        otherdecode_bitstreams_size.push_back(img_len[i]);
        otherdecode_output.push_back(out[i]);
      }
    }
  } else {
    for(int i = 0; i < params.batch_size; i++) {
      otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
      otherdecode_bitstreams_size.push_back(img_len[i]);
      otherdecode_output.push_back(out[i]);
    }
  }

  CHECK_CUDA(cudaEventRecord(startEvent, params.stream));

    if(batched_bitstreams.size() > 0)
     {

       /*
       CHECK_NVJPEG(
               nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                                            batched_bitstreams.size(), 1, params.fmt));

         CHECK_NVJPEG(nvjpegDecodeBatched(
             params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.data(),
             batched_bitstreams_size.data(), batched_output.data(), params.stream)); */

       std::cout << "Doing DECOUPLED STATE decodification" << std::endl;

	 // This is the first stage of the decoupled decoding process.
	 // It is done entirely on the host, hence it is synchronous with respect of the host. 
	 nvjpegDecodeJpegHost(
           params.nvjpeg_handle, params.nvjpeg_decoder,
	   params.nvjpeg_decoupled_state,
	   params.nvjpeg_decode_params, params.jpeg_streams[0]);

	 CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(
           params.nvjpeg_decoupled_state, params.device_buffer));

	nvjpegDecodeJpegTransferToDevice(
           params.nvjpeg_handle, params.nvjpeg_decoder,
	   params.nvjpeg_decoupled_state,
	   params.jpeg_streams[0], params.stream);

        nvjpegDecodeJpegDevice(
           params.nvjpeg_handle, params.nvjpeg_decoder,
	   params.nvjpeg_decoupled_state,
	   batched_output.data(), params.stream);		
     }

    if(otherdecode_bitstreams.size() > 0)
    {
          CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
          int buffer_index = 0;
          CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
          for (int i = 0; i < params.batch_size; i++) {
              CHECK_NVJPEG(
                  nvjpegJpegStreamParse(params.nvjpeg_handle, otherdecode_bitstreams[i], otherdecode_bitstreams_size[i],
                  0, 0, params.jpeg_streams[buffer_index]));

              CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
                  params.pinned_buffers[buffer_index]));

              CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.nvjpeg_decode_params, params.jpeg_streams[buffer_index]));

              CHECK_CUDA(cudaStreamSynchronize(params.stream));

              CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.jpeg_streams[buffer_index], params.stream));

              buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

              CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  &otherdecode_output[i], params.stream));


          }
    }
  CHECK_CUDA(cudaEventRecord(stopEvent, params.stream));

  CHECK_CUDA(cudaEventSynchronize(stopEvent));
  CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
  time = static_cast<double>(loopTime);


  //   Metric_Calculator<<<1, 1>>>(&otherdecode_output[0]);
   my_kernel<<<1,1>>>(img_info.channel[0], img_info.channel[1], img_info.channel[2]);
   cudaDeviceSynchronize();
   
  return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, decode_params_t &params,
                      double &total) {
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames current_names(params.batch_size);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);
  // we wrap over image files to process total_images of files
  FileNames::iterator file_iter = image_names.begin();

  // stream for decoding
  CHECK_CUDA(
      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  int total_processed = 0;

  // output buffers
  std::vector<nvjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);

  for (int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }

  double test_time = 0;
  int warmup = 0;
  while (total_processed < params.total_images) {
    if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                        file_len, current_names))
      return EXIT_FAILURE;

    if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                        current_names, params))
      return EXIT_FAILURE;

    double time;
    if (decode_images(file_data, file_len, iout, params, time))
      return EXIT_FAILURE;
    if (warmup < params.warmup) {
      warmup++;
    } else {
      total_processed += params.batch_size;
      test_time += time;
    }

    //if (params.write_decoded)
    // write_images(iout, widths, heights, params, current_names);
  }
  total = test_time;

  release_buffers(iout);

  CHECK_CUDA(cudaStreamDestroy(params.stream));

  return EXIT_SUCCESS;
}
