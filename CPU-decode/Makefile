all:
	/home/users/bsalves/CUDA/install/bin/nvcc -c -g -Xcompiler -fopenmp -std=c++17 -O2 -ljpeg -lnvToolsExt jpeg_image.cpp
	/home/users/bsalves/CUDA/install/bin/nvcc -g -Xcompiler -fopenmp -lgomp jpeg_image.o cpu-decode.cpp -std=c++17 -O2 -ljpeg -lnvToolsExt -o cpu-decode
#	g++ -c -std=c++17 -O2 -ljpeg jpeg_image.cpp
#	g++ jpeg_image.o cpu-decode.cpp -std=c++17 -O2 -ljpeg -o cpu-decode
clean:
	rm cpu-decode
