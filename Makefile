crop_compile: bmp_crop.c
	gcc -o crop bmp_crop.c 

crop: crop_compile
	./crop image.bmp scan.bmp 100 100 200 200

cuda_scanner: strict_scanner.cu
	nvcc -o cuda_scanner strict_scanner.cu

c_scanner: strict_scanner.c
	gcc -o c_scanner strict_scanner.c

run_c: c_scanner
	./c_scanner image.bmp scan.bmp

run_cu: cuda_scanner
	./cuda_scanner image.bmp scan.bmp

clean:
	rm -f crop cuda_scanner c_scanner
