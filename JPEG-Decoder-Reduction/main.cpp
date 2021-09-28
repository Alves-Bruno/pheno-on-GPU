#include "JPEG-Decoder-Reduction.h"

int main(int argc, const char *argv[]) {

  decode_params_t params;
  
  params.input_dir = "./input_images/img1.jpg";
  params.batch_size = 1;
  params.total_images = 1;
  params.warmup = 0;
  params.fmt = NVJPEG_OUTPUT_RGB;
  params.write_decoded = false;

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};

  nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator,
                                &pinned_allocator,NVJPEG_FLAGS_DEFAULT,  &params.nvjpeg_handle);
  params.hw_decode_available = true;
  if( status == NVJPEG_STATUS_ARCH_MISMATCH) {
    std::cout<<"Hardware Decoder not supported. Falling back to default backend"<<std::endl;
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                              &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle));
    params.hw_decode_available = false;
  } else {
    CHECK_NVJPEG(status);
  }

  CHECK_NVJPEG(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

  create_decoupled_api_handles(params);

  // read source images
  FileNames image_names;
  // readInput(params.input_dir, image_names);
  image_names.push_back(params.input_dir);
  
  double total;
  if (process_images(image_names, params, total)) return EXIT_FAILURE;
  std::cout << "Total decoding time: " << total << std::endl;
  std::cout << "Avg decoding time per image: " << total / params.total_images
            << std::endl;
  std::cout << "Avg images per sec: " << params.total_images / total
            << std::endl;
  std::cout << "Avg decoding time per batch: "
            << total / ((params.total_images + params.batch_size - 1) /
                        params.batch_size)
            << std::endl;

  destroy_decoupled_api_handles(params);

  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}
