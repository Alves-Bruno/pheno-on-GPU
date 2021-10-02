all:
	g++ -c -std=c++17 -ljpeg jpeg_image.cpp
	g++ jpeg_image.o cpu-decode.cpp -std=c++17 -ljpeg -o cpu-decode
clean:
	rm cpu-decode
