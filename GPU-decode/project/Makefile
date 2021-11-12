all:
	/home/users/bsalves/CUDA/install/bin/nvcc -gencode arch=compute_61,code=sm_61 -g -Xcompiler -fopenmp main.cpp gpu_decode.cu -lnvjpeg -lnvToolsExt -o run
#	/home/users/bsalves/CUDA/install/bin/nvcc -g stream-decode.cu -lnvjpeg -lnvToolsExt -o run
## For hype4:
# SET(CMAKE_CUDA_FLAGS "-gencode arch=compute_37,code=sm_37")
# For tupi1:
# SET(CMAKE_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61")

clean:
	rm run
