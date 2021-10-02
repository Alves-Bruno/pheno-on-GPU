#include <stdio.h>
#include <jpeglib.h>
#include <sys/time.h>
#include <math.h>
#include <string>
#include "jpeg_image.h"
#include <chrono>

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

std::vector<rgb> phenovis_rgb_mean(std::vector<std::string> images)
{

  // names is a vector to keep image names
  std::vector<std::string> names;
  std::vector<rgb> avg_values;

  int i, row_number = 0;
  for (i = 0; i < images.size(); i++) {
   
    // Load the image and apply mask
    double decode_time;
    image_t *image = load_jpeg_image_with_time(images[i].c_str(), &decode_time);
    
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

      printf("%s, %f, %f, %f, %lf, %lf\n", images[i].c_str(), r_avg, g_avg, b_avg, decode_time,calc_time);

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

int main(int argc, char **argv){

  std::vector<std::string> input_images;
  input_images.push_back("x.jpg");
  
  phenovis_rgb_mean(input_images);
  
}
