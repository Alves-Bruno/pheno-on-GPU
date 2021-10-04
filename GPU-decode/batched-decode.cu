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

void malloc_check(void *ptr, const char *message){
  if(ptr == NULL){
    std::cout << "Could not malloc [" << ptr << "]" << std::endl;
    exit(1);
  }
}
int main(int argc, char *argv[]){

  if(argc < 4){
    std::cout << "Usage: " << argv[0] <<
      " <images_dir> <batch_size> <number_of_images>" << std::endl;
    exit(1);
  }
  std::vector<std::string> input_images_names;
  const std::string images_path = argv[1];

  int batch_size = atoi(argv[2]);
  int total_images = atoi(argv[3]);
  //std::cout << "batch_size: " << batch_size << ", total_images: " << total_images << std::endl;
  
  // Read images at path
  readInput(images_path, input_images_names);

  if(total_images > input_images_names.size())
    total_images = input_images_names.size();

  if(batch_size > total_images)
    batch_size = total_images;
    
  // Memory allocation 
  unsigned char **input_images_buffer = (unsigned char**) malloc(total_images * sizeof(unsigned char *));
  malloc_check((void*) input_images_buffer, "input_images_buffer");
  size_t *input_images_buffer_sizes = (size_t*) malloc(total_images * sizeof(size_t));
  malloc_check((void*)input_images_buffer_sizes, "input_images_buffer_sizes");

  auto start_fread = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < total_images; i++){
    //std::cout << i << " " << input_images_names[i] << std::endl;
    FILE* file = fopen(input_images_names[i].c_str(), "rb");
    fseek(file, 0, SEEK_END);
    unsigned long size=ftell(file);

    unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
    malloc_check((void*) jpg, "jpg");
    fseek(file, 0, 0);
    fread(jpg, size, 1, file);

    fclose(file);
    //free(file);

    input_images_buffer[i] = jpg;
    input_images_buffer_sizes[i] = (size_t) size;
  }
  auto end_fread = std::chrono::high_resolution_clock::now();

  //  std::cout << "GPU starts" << std::endl;
  
  nvjpegHandle_t handle;
  nvjpegCreateSimple(&handle);
  nvjpegJpegState_t jpeg_handle;
  nvjpegJpegStateCreate(handle, &jpeg_handle);

  nvjpegDecodeBatchedInitialize(handle, jpeg_handle,
                                batch_size, 1, NVJPEG_OUTPUT_RGB);

  int n_rounds = total_images / batch_size;
  nvjpegImage_t **dest_handle = (nvjpegImage_t**) malloc(n_rounds * sizeof(nvjpegImage_t*));
  malloc_check((void*)dest_handle, "dest_handle");

  int *considered_pixels = (int*) malloc(sizeof(int) * total_images);

  int round_i = 0;
  for(int b_start = 0; b_start < total_images; b_start += batch_size){

    // Malloc the output buffer
    nvjpegImage_t *destinations = (nvjpegImage_t*) malloc(batch_size * sizeof(nvjpegImage_t));
    malloc_check((void*) destinations, "destinations");
    int dest_i = 0;
    for(int i = b_start; i < b_start + batch_size; i++){
      int nComponents, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
      nvjpegChromaSubsampling_t subsampling;
      nvjpegGetImageInfo(
			 handle,
			 input_images_buffer[i],
			 input_images_buffer_sizes[i],
			 &nComponents, &subsampling, widths, heights);
      
      nvjpegImage_t img_info;
      for(int c=0; c<3; c++){
	img_info.pitch[c] = widths[0];
	if(cudaMalloc((void**)&img_info.channel[c], widths[0]*heights[0]) != cudaSuccess){
	  std::cout << "CudaMalloc error" << std::endl;
	  exit(1);
	}
      }

      considered_pixels[i] = widths[0]*heights[0];
      destinations[dest_i] = img_info;
      dest_i++;
    }
    
    dest_handle[round_i] = destinations;
    round_i++;

  }

  cudaStream_t *streams = (cudaStream_t*) malloc(n_rounds * sizeof(cudaStream_t));
  auto start_decode = std::chrono::high_resolution_clock::now();
  round_i = 0;
  for(int b_start = 0; b_start < total_images; b_start += batch_size){

    cudaStreamCreateWithFlags(&streams[round_i], cudaStreamNonBlocking);
    nvjpegDecodeBatched(
			handle,
			jpeg_handle,
			&input_images_buffer[b_start],
			&input_images_buffer_sizes[b_start],
			dest_handle[round_i],
			streams[round_i]);

    round_i++;
  }
  cudaDeviceSynchronize();
  auto end_decode = std::chrono::high_resolution_clock::now();

  auto start_calc = std::chrono::high_resolution_clock::now();
  int *channel_avg = (int *) malloc(total_images * 3 * sizeof(int));
  for(int i = 0; i < total_images; i++){
    for(int c=0; c<3; c++){

      //      std::cout << c << " " << i/batch_size << " " << i%batch_size << std::endl;
      //printf("[%d, %d]: dest_handle[%d][%d]\n", i, c,  i/batch_size,  i%batch_size);
      thrust::device_ptr<unsigned char> channel((unsigned char*) dest_handle[i/batch_size][i%batch_size].channel[c]);
      
      channel_avg[(i*3) + c] = thrust::reduce(
          channel, // Vector start
	  channel + considered_pixels[i], // Vector end 
	  (int) 0, // reduce first value
	  thrust::plus<int>()); // reduce operation

      //std::cout << "AVG: " << channel_avg[(i*3) + c] / (float) considered_pixels[i] << std::endl;
    }
  }
  cudaDeviceSynchronize();
  auto end_calc = std::chrono::high_resolution_clock::now();
  
  double fread_time = std::chrono::duration_cast<std::chrono::microseconds>(end_fread - start_fread).count();
  double decode_time = std::chrono::duration_cast<std::chrono::microseconds>(end_decode - start_decode).count();
  double calc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_calc - start_calc).count();
  //  printf("fread_time, decode_time, fread_time.by_image, decode_time.by_image\n");
  printf("%lf, %lf, %lf, %lf, %lf, %lf\n",
	 fread_time,
	 decode_time,
	 calc_time,
	 fread_time / (float)total_images,
	 decode_time / (float)total_images,
	 calc_time / (float)total_images
	 );

  //int image_i = 0;
  //printf("[%d] R %d G %d B %d\n", image_i, channel_avg[(image_i*3) + 0], channel_avg[(image_i*3) + 1], channel_avg[(image_i*3) + 2]);
  //image_i = total_images - 1;
  //printf("[%d] R %d G %d B %d\n", image_i, channel_avg[(image_i*3) + 0], channel_avg[(image_i*3) + 1], channel_avg[(image_i*3) + 2]);
  
  // Free all the stuff
  for(int i = 0; i < total_images; i++){
    free(input_images_buffer[i]);
  }
  free(input_images_buffer);
  free(input_images_buffer_sizes);
  for(int i = 0; i < n_rounds; i++){
    free(dest_handle[i]);
    cudaStreamDestroy(streams[i]);
  }
  free(dest_handle);
  free(considered_pixels);
  
  /*
  unsigned char *red_char_host = (unsigned char*) malloc(sizeof(unsigned char) * (widths[0]*heights[0]));
  cudaError_t copy_err = cudaMemcpy((void*)red_char_host, (void*)destinations[0].channel[0], (widths[0]*heights[0]) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
  
  if (copy_err == cudaSuccess){
    printf("COPY SUCCESS\n");
    int sum_check = 0;
    for(int i=0; i<(widths[0]*heights[0]); i++){
      //printf("[%d]: %d\n", i, red_char_host[i]);
      sum_check += (int) red_char_host[i];
    }
    printf("RED SUM CHECK: %d\n", sum_check);
    printf("RED SUM CHECK: %f\n", sum_check / (float)(widths[0]*heights[0]));
    
    }
  */
    
  /*
  int channel_sum[3];
  for(int i=0; i<3; i++){
    
    thrust::device_ptr<unsigned char> channel((unsigned char*)img_info.channel[i]);
    //    thrust::device_ptr<unsigned char> channel((unsigned char*)out_buffer[0].channel[i]);
    // compute sum on the device
    channel_sum[i] = thrust::reduce(
      channel, // Vector start
      channel + (widths[0]*heights[0]), // Vector end 
      (int) 0, // reduce first value
      thrust::plus<int>()); // reduce operation
    
  }
  cudaDeviceSynchronize();
    
  for(int i=0; i<3; i++){
    printf("Channel[%d] sum: %d\n", i, channel_sum[i]);
    printf("Channel[%d] avg: %f\n", i, channel_sum[i] / (float)(widths[0]*heights[0]));
  }

  double fread_time = std::chrono::duration_cast<std::chrono::microseconds>(end_fread - start_fread).count();
  double decode_time = std::chrono::duration_cast<std::chrono::microseconds>(end_decode - start_decode).count();
  double calc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_calc - start_calc).count();

  printf("%s, %lf, %lf, %lf\n", argv[1], fread_time, decode_time, calc_time);
  */
  
}
