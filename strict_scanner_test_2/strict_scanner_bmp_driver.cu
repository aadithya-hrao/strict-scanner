#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>

#include "libstrictscanner.h"

#pragma pack(1)
typedef struct {
    unsigned short bfType;
    unsigned int   bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int   bfOffBits;
} BITMAPFILEHEADER;

#pragma pack(1)
typedef struct {
    unsigned int   biSize;
    int            biWidth;
    int            biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    int            biXPelsPerMeter;
    int            biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
} BITMAPINFOHEADER;

// Threshold for matching
// 1.0 means that the token must overlay perfectly on 
// that particular part of the image on which it is being overlayed
float SIMILARITY_THRESHOLD = 1.0;

int setup_bmp(char* bmp_image_file_path, uint8_t** image, uint32_t* image_dimx, uint32_t* image_dimy, int* fd, void** mapped, struct stat* sb) {
    // Open the file
    *fd = open(bmp_image_file_path, O_RDONLY);
    if (*fd == -1) {
        perror("Error opening file");
        return 1;
    }

    if (fstat(*fd, sb) == -1) {
        perror("Error getting file size");
        return 1;
    }

    // Map the file into memory
    *mapped = mmap(NULL, sb->st_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*mapped == MAP_FAILED) {
        perror("Error mapping file");
        return 1;
    }

    // Interpret the headers
    BITMAPFILEHEADER *bmfh = (BITMAPFILEHEADER *)*mapped;
    BITMAPINFOHEADER *bmih = (BITMAPINFOHEADER *)((char *)*mapped + sizeof(BITMAPFILEHEADER));

    // Check if it's an 8-bit grayscale bitmap
    if (bmih->biBitCount == 8 && bmih->biCompression == 0) {
        // WARNING: We treat the height as the x dimension and the width as the y dimension
        *image_dimx = bmih->biHeight;
        *image_dimy = bmih->biWidth;
        *image = (uint8_t *)((char *)*mapped + bmfh->bfOffBits);

        //copy the image to image
    } else {
        printf("The image is not an 8-bit grayscale bitmap.\n");
        return 1;
    }

    return 0;
}

int unsetup_bmp(int fd, void* mapped, struct stat* sb) {
    // Clean up
    munmap(mapped, sb->st_size);
    close(fd);

    return 0;
}

int main(int argc, char *argv[]) {
    // get the file path
    if (argc != 3) {
        printf("Usage: %s <bmp_image_file_path> <bmp_scan_file_path>\n", argv[0]);
        return 1;
    }
    char *bmp_image_file_path = argv[1];
    char *bmp_scan_file_path = argv[2];

    // setup the image
    uint8_t *image;
    uint32_t image_dimx, image_dimy;
    int image_fd;
    void *image_mapped;
    struct stat image_sb;
    if (setup_bmp(bmp_image_file_path, &image, &image_dimx, &image_dimy, &image_fd, &image_mapped, &image_sb)) {
        printf("Error setting up the image.\n");
        return 1;
    }
    printf("Image dimensions: %d x %d (H*W) (M*N) (X*Y)\n", image_dimx, image_dimy);

    // setup the scan
    uint8_t *scan;
    uint32_t scan_dimx, scan_dimy;
    int scan_fd;
    void *scan_mapped;
    struct stat scan_sb;
    if (setup_bmp(bmp_scan_file_path, &scan, &scan_dimx, &scan_dimy, &scan_fd, &scan_mapped, &scan_sb)) {
        printf("Error setting up the scan.\n");
        unsetup_bmp(image_fd, image_mapped, &image_sb);
        return 1;
    }
    printf("Scan dimensions: %d x %d (H*W) (M*N) (X*Y)\n", scan_dimx, scan_dimy);

    if (scan_dimx > image_dimx || scan_dimy > image_dimy) {
        printf("The scan is larger than the image.\n");
        unsetup_bmp(scan_fd, scan_mapped, &scan_sb);
        unsetup_bmp(image_fd, image_mapped, &image_sb);
        return 1;
    }

    bool *match_matrix = (bool *)malloc(image_dimx * image_dimy * sizeof(bool));
    for (uint32_t x = 0; x < image_dimx; x++) {
        for (uint32_t y = 0; y < image_dimy; y++) {
            match_matrix[x * image_dimy + y] = false;
        }
    }

    // scan the image
    printf("Scanning with threshold %f\n", SIMILARITY_THRESHOLD);
    int result = strict_scan(image, image_dimx, image_dimy, scan, scan_dimx, scan_dimy, SIMILARITY_THRESHOLD, match_matrix);
    unsetup_bmp(scan_fd, scan_mapped, &scan_sb);
    unsetup_bmp(image_fd, image_mapped, &image_sb);

    if (result) {
        printf("Error scanning the image.\n");
        return 1;
    }

    // print offsets in image if match
    printf("Matches:\n");
    for (uint32_t x = 0; x < image_dimx; x++) {
        for (uint32_t y = 0; y < image_dimy; y++) {
            if (match_matrix[x * image_dimy + y]) {
                printf("(%d, %d)", x, y);
            }
        }
    }

    return 0;
}

// WARNING: If you create your own bitmap, and then a cropped version of it
// make sure to do it in the right software, or you'll get the wrong results.

// Yes, this file seems to HAVE to have a CUDA extension as I'm linking it to another CUDA file,
// even though this file doesn't have any CUDA code in it.
