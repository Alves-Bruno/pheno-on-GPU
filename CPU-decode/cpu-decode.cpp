#include <stdio.h>
#include <jpeglib.h>
#include <sys/time.h>
#include <math.h>
#include <string>
#include "jpeg_image.h"
#include <chrono>

#include <stdlib.h>
#include <iostream>
#include <vector>

#include <dirent.h>  
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>

#include <nvToolsExt.h> 
#include <sys/syscall.h>
#include <unistd.h>

#include <omp.h>


typedef struct {
    unsigned char r;       // a fraction between 0 and 1
    unsigned char g;       // a fraction between 0 and 1
    unsigned char b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    int r;       // a fraction between 0 and 1
    int g;       // a fraction between 0 and 1
    int b;       // a fraction between 0 and 1
} rgb_sum;

rgb get_rgb_for_pixel_256(int pixel, image_t *image) {
  unsigned char *iimage = image->image;
  unsigned char r = iimage[pixel];
  unsigned char g = iimage[pixel + 1];
  unsigned char b = iimage[pixel + 2];

  rgb RGB = {r, g, b};
  return RGB;
}

void phenovis_rgb_mean(
    std::vector<std::string> &images,
    std::vector<unsigned int> &images_id,
    std::vector<float> &avg_values,
    int total_images,
    bool show_rgb)
{

  if(show_rgb)
    printf("image_id, x, y, R, G, B\n");
  
#pragma omp parallel default(shared)
{
#pragma omp for schedule(static)
  for(int i = 0; i < total_images; i++) {
   
    // Load the image and apply mask
    double decode_time;
    nvtxRangePush("HOST_decode");
    image_t *image = load_jpeg_image_with_time(images[i].c_str(), &decode_time);
    nvtxRangePop();
    
    int considered_pixels = image->width * image->height;
    //if (global_mask) {
    //  considered_pixels = apply_mask(image, global_mask);
    //}

    //printf("Image size: %d x %d\n", image->width, image->height);

    nvtxRangePush("avg_calc");

      int count_pixels = 0;
      
      rgb_sum rgb_sum_;
      rgb_sum_.r = 0;
      rgb_sum_.g = 0;
      rgb_sum_.b = 0;
      
    // For every pixel...
      for (int p = 0; p < image->size; p += 3) {

        rgb RGB = get_rgb_for_pixel_256(p, image);
        rgb_sum_.r += RGB.r;
        rgb_sum_.g += RGB.g;
        rgb_sum_.b += RGB.b;

	if(show_rgb){
	  printf("%d, %d, %d, %d, %d, %d\n", i, count_pixels / image->height,
		 count_pixels % image->width, RGB.r, RGB.g, RGB.b);
	}

        count_pixels += 1;
      
      } //printf("\n");
       
      float r_avg = rgb_sum_.r / (float) count_pixels;
      float g_avg = rgb_sum_.g / (float) count_pixels;
      float b_avg = rgb_sum_.b / (float) count_pixels;

      //printf("%s, %f, %f, %f, %lf, %lf\n", images[i].c_str(), r_avg, g_avg, b_avg, decode_time,calc_time);

      #pragma omp critical
      {
      images_id.push_back(i);
      avg_values.push_back(r_avg);
      avg_values.push_back(g_avg);
      avg_values.push_back(b_avg);
      }

      nvtxRangePop();
      
      //Free the image data
      free(image->image);
      free(image);
  }
}
  
  //return(sum_values);

}


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

void pre_load(std::vector<std::string> &images,
	      int total_images,
	      unsigned char **buffer_images, size_t *buffer_images_sizes){

  nvtxRangePush(__FUNCTION__);

#pragma omp parallel default(shared)
{
#pragma omp for schedule(static)
  //  std::vector<image_file> images_buffer;
  for(int i = 0; i < total_images; i++){

    FILE* file = fopen(images[i].c_str(), "rb");
    fseek(file, 0, SEEK_END);
    unsigned long size=ftell(file);

    unsigned char* jpg = (unsigned char*)malloc(size * sizeof(char));
    fseek(file, 0, 0);
    fread(jpg, size, 1, file);

    fclose(file);

    #pragma omp atomic
    buffer_images[i] = jpg;
    #pragma omp atomic
    buffer_images_sizes[i] = (size_t) size;
  }
}  
  nvtxRangePop();
}

int main(int argc, char **argv){

  nvtxRangePush(__FUNCTION__);
  if(argc < 3){
    std::cout << "Usage: " << argv[0] <<
      " <images_dir> <number_of_images>" << std::endl;
    exit(1);
  }
  
  std::vector<std::string> input_images_names;
  const std::string images_path = argv[1];

  int total_images = atoi(argv[2]);
  
  // Read images at path
  readInput(images_path, input_images_names);

  double total_calc_time = 0;
  double total_decode_time = 0;
  std::vector<float> avg_values;
  std::vector<unsigned int> images_id;

  nvtxRangePush("malloc_input_buffer");
  unsigned char **buffer_images = (unsigned char**) malloc(input_images_names.size()
							   * sizeof(unsigned char *));
  size_t *buffer_images_sizes = (size_t*) malloc(input_images_names.size() * sizeof(size_t));
  nvtxRangePop();
  
  pre_load(input_images_names, total_images, buffer_images, buffer_images_sizes);
  nvtxRangePush("free_buffer");
  for(int i=0; i < total_images; i++){
    free(buffer_images[i]);
  }
  free(buffer_images);
  free(buffer_images_sizes);
  nvtxRangePop();

  phenovis_rgb_mean(input_images_names, images_id, avg_values, total_images, false);

  bool show_avg = false;
  if(show_avg){

    printf("image_id, image_name, R, G, B\n");
    for(int i = 0; i < total_images; i++){
      printf("%d, %s, %f, %f, %f\n", images_id[i], input_images_names[i].c_str(), 
	     avg_values[(i*3)+0],
	     avg_values[(i*3)+1],
	     avg_values[(i*3)+2]);
    }
  }
  
  nvtxRangePop();
  
}

