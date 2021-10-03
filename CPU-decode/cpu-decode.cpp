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

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

rgb get_rgb_for_pixel_256(int pixel, image_t *image) {
  unsigned char *iimage = image->image;
  unsigned char r = iimage[pixel];
  unsigned char g = iimage[pixel + 1];
  unsigned char b = iimage[pixel + 2];

  rgb RGB = {(double)r, (double)g, (double)b};
  return RGB;
}

std::vector<rgb> phenovis_rgb_mean(
    std::vector<std::string> images,
    int total_images,
    double *total_decode_time, double *total_calc_time)
{

  // names is a vector to keep image names
  std::vector<std::string> names;
  std::vector<rgb> avg_values;

  //  double total_calc_time = 0;
  //  double total_decode_time = 0;

  int i, row_number = 0;
  //  for (i = 0; i < images.size(); i++) {
  for (i = 0; i < total_images; i++) {
   
    // Load the image and apply mask
    double decode_time;
    image_t *image = load_jpeg_image_with_time(images[i].c_str(), &decode_time);
    *total_decode_time += decode_time;
    
    int considered_pixels = image->width * image->height;
    //if (global_mask) {
    //  considered_pixels = apply_mask(image, global_mask);
    //}

      auto start_calc = std::chrono::high_resolution_clock::now();

      double count_pixels = 0;
      double r_sum = 0;
      double g_sum = 0;
      double b_sum = 0;
      
    // For every pixel...
      for (int p = 0; p < image->size; p += 3) {

        rgb RGB = get_rgb_for_pixel_256(p, image);
        r_sum += RGB.r;
        g_sum += RGB.g;
        b_sum += RGB.b;
        count_pixels += 1;

      }
       
      double r_avg = r_sum / (float) count_pixels;
      double g_avg = g_sum / (float) count_pixels;
      double b_avg = b_sum / (float) count_pixels;

      auto end_calc = std::chrono::high_resolution_clock::now();
      double calc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_calc - start_calc).count();
      *total_calc_time += calc_time;

      //printf("%s, %f, %f, %f, %lf, %lf\n", images[i].c_str(), r_avg, g_avg, b_avg, decode_time,calc_time);

      // Push back the image names
      names.push_back(images[i]);
      rgb rgb_avg;
      rgb_avg.r = r_avg;
      rgb_avg.g = g_avg;
      rgb_avg.b = b_avg;
      avg_values.push_back(rgb_avg);
      
      //Free the image data
      free(image->image);
      free(image);
  }

  
  return(avg_values);

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


int main(int argc, char **argv){

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
  phenovis_rgb_mean(input_images_names, total_images, &total_decode_time, &total_calc_time);

  //printf("decode_time, calc_time, decode_time.by_image, calc_time.by_image\n");
  printf("%lf, %lf, %lf, %lf\n",
	 total_decode_time,
	 total_calc_time,
	 total_decode_time / (float)total_images,
	 total_calc_time / (float)total_images);
  
}

