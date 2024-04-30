#!/usr/bin/make -f

####################

bmp_crop: bmp_crop.c
	gcc -o bmp_crop bmp_crop.c

crop: bmp_crop
# change following values to crop the image
	./bmp_crop lena_gray.bmp scan10.bmp 100 100 200 200

###################

# compile the strict scanner (cuda)
strict_scanner_bmp_driver_cuda: strict_scanner_bmp_driver.cu libstrictscanner_SRN.cu
	nvcc -o strict_scanner_bmp_driver $?

# compile the strict scanner (serial)
strict_scanner_bmp_driver_serial: strict_scanner_bmp_driver.cu libstrictscanner_SRN_serial.cu
	nvcc -o strict_scanner_bmp_driver $?

# run the strict scanner (whichever is compiled)
run_bmp_driver:
	./strict_scanner_bmp_driver lena_gray.bmp scan.bmp

###################

# compile the strict scanner (cuda)
strict_scanner_basic_test_driver_cuda: strict_scanner_basic_test_driver.cu libstrictscanner_SRN.cu
	nvcc -o strict_scanner_basic_test_driver $?

# compile the strict scanner (serial)
strict_scanner_basic_test_driver_serial: strict_scanner_basic_test_driver.cu libstrictscanner_SRN_serial.cu
	nvcc -o strict_scanner_basic_test_driver $?

# run the strict scanner (whichever is compiled)
run_basic_test_driver:
	./strict_scanner_basic_test_driver

###################

# compile the strict scanner (cuda)
strict_scanner_big_run_driver_cuda: strict_scanner_big_run_driver.cu libstrictscanner_SRN.cu
	nvcc -o strict_scanner_big_run_driver $?

# compile the strict scanner (serial)
strict_scanner_big_run_driver_serial: strict_scanner_big_run_driver.cu libstrictscanner_SRN_serial.cu
	nvcc -o strict_scanner_big_run_driver $?

# run the strict scanner (whichever is compiled)
run_big_run_driver:
	# ./strict_scanner_big_run_driver
	time -vvv ./strict_scanner_big_run_driver
