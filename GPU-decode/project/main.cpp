#include "gpu_decode.h"

int readInput(const std::string &sInputPath, std::vector<std::string> &filelist);

int main(int argc, char *argv[]){
  nvtxRangePush(__FUNCTION__);

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
  float* avg_values = (float*) malloc(sizeof(float) * total_images * 3);
  malloc_check(avg_values, "avg_values");
  decoder.decode_and_calculate(avg_values, false);
  //decoder.decode_decoupled();

  //  for(int i = 0; i < input_images_names.size(); i++){
  //  std::cout << input_images_names[i] << std::endl;
  //}
  
  //for(int i = 0; i < avg_values.size(); i++){
  //  std::cout << avg_values[i] << std::endl;
  //}
  
  bool show_avg = false;
  if(show_avg){

    printf("image_id, image_name, R, G, B\n");
    for(int i = 0; i < total_images; i++){
      printf("%d, %s, %f, %f, %f\n", i, input_images_names[i].c_str(), 
	     avg_values[(i*3)+0],
	     avg_values[(i*3)+1],
	     avg_values[(i*3)+2]);
    }
  }

  cudaDeviceSynchronize();
  return(0);

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
    nvtxRangePop();
    return 0;
}
