// #include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

auto block_threads = dim3(16, 16);

__global__ void check_token(uint8_t *image, uint32_t image_dimx,
                            uint32_t image_dimy, uint8_t *token,
                            uint32_t token_dimx, uint32_t token_dimy,
                            float error_threshold, 
                            bool *match_matrix) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > (image_dimx - token_dimx + 1) || y > (image_dimy - token_dimy + 1))
    return;


  float match_difference_error = 0;

  for (uint32_t tx = 0; tx < token_dimx; tx++) {
    for (uint32_t ty = 0; ty < token_dimy; ty++) {
      match_difference_error += abs(image[(x + tx) * image_dimy + (y + ty)] -
                                    token[tx * token_dimy + ty]);
    }
  }
  if (match_difference_error <= error_threshold) {
    match_matrix[x * image_dimy + y] = true;
  }
}

__global__ void reduce_mat(bool *matrix, uint32_t image_dimx,
                           uint32_t image_dimy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > image_dimx || y > image_dimy)
    return;
  if (matrix[x * image_dimx + y] == true) {
    printf("x: %d, y: %d\n", x, y);
  }
}
#pragma pack(1)
typedef struct {
  unsigned short bfType;
  unsigned int bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned int bfOffBits;
} BITMAPFILEHEADER;

#pragma pack(1)
typedef struct {
  unsigned int biSize;
  int biWidth;
  int biHeight;
  unsigned short biPlanes;
  unsigned short biBitCount;
  unsigned int biCompression;
  unsigned int biSizeImage;
  int biXPelsPerMeter;
  int biYPelsPerMeter;
  unsigned int biClrUsed;
  unsigned int biClrImportant;
} BITMAPINFOHEADER;

// Threshold for matching
// 1.0 means that the token must overlay perfectly on
// that particular part of the image on which it is being overlayed
float SIMILARITY_THRESHOLD = 1.0;

// Strict scan that compares the token with the image pixel by pixel
// WARNING: Note that X*Y is basically an M*N matrix, where
// M (X) is the height/vertical and N (Y) is the width/horizontal
int strict_scan(uint8_t *image, uint32_t image_dimx, uint32_t image_dimy,
                uint8_t *token, uint32_t token_dimx, uint32_t token_dimy,
                float match_similarity_threshold, bool *match_matrix) {
  // accuracy increases by multiplying here (instead of dividing the final value
  // later)
  float error_threshold =
      (1 - match_similarity_threshold) * 255 * token_dimy * token_dimx;
  printf("Error threshold: %f\n", error_threshold);

  auto blocks = dim3(((image_dimx - token_dimx + 1) + block_threads.x-1) / block_threads.x,
                     ((image_dimy - token_dimy + 1) + block_threads.y-1) / block_threads.y);


  check_token<<<blocks, block_threads>>>(image, image_dimx, image_dimy, token,
                                   token_dimx, token_dimy,
                                   match_similarity_threshold, match_matrix);

  return 0;
}

int setup_bmp(char *bmp_image_file_path, uint8_t **image, uint32_t *image_dimx,
              uint32_t *image_dimy, int *fd, void **mapped, struct stat *sb) {
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
  BITMAPINFOHEADER *bmih =
      (BITMAPINFOHEADER *)((char *)*mapped + sizeof(BITMAPFILEHEADER));

  // Check if it's an 8-bit grayscale bitmap
  if (bmih->biBitCount == 8 && bmih->biCompression == 0) {
    // WARNING: We treat the height as the x dimension and the width as the y
    // dimension
    *image_dimx = bmih->biHeight;
    *image_dimy = bmih->biWidth;
    *image = (uint8_t *)((char *)*mapped + bmfh->bfOffBits);

    // copy the image to image
  } else {
    printf("The image is not an 8-bit grayscale bitmap.\n");
    return 1;
  }

  return 0;
}

int unsetup_bmp(int fd, void *mapped, struct stat *sb) {
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
  if (setup_bmp(bmp_image_file_path, &image, &image_dimx, &image_dimy,
                &image_fd, &image_mapped, &image_sb)) {
    printf("Error setting up the image.\n");
    return 1;
  }
  printf("Image dimensions: %d x %d (H*W) (M*N) (X*Y)\n", image_dimx,
         image_dimy);

  // setup the scan
  uint8_t *scan;
  uint32_t scan_dimx, scan_dimy;
  int scan_fd;
  void *scan_mapped;
  struct stat scan_sb;
  if (setup_bmp(bmp_scan_file_path, &scan, &scan_dimx, &scan_dimy, &scan_fd,
                &scan_mapped, &scan_sb)) {
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

  bool *d_match_matrix;
  cudaMalloc(&d_match_matrix, (image_dimx * image_dimy));

  // scan the image
  printf("Scanning with threshold %f\n", SIMILARITY_THRESHOLD);
  uint8_t *d_image;
  uint8_t *d_scan;

  cudaMalloc(&d_image, (image_dimx * image_dimy));
  cudaMalloc(&d_scan, (scan_dimx * scan_dimy));

  cudaMemcpy(d_image, image, image_dimx * image_dimy, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scan, scan, scan_dimx * scan_dimy, cudaMemcpyHostToDevice);

  int result = strict_scan(d_image, image_dimx, image_dimy, d_scan, scan_dimx,
                           scan_dimy, SIMILARITY_THRESHOLD, d_match_matrix);
  unsetup_bmp(scan_fd, scan_mapped, &scan_sb);
  unsetup_bmp(image_fd, image_mapped, &image_sb);

  if (result) {
    printf("Error scanning the image.\n");
    return 1;
  }

  // print offsets in image if match

  printf("Matches:\n");
  uint32_t *res_x, *res_y;
  cudaMalloc(&res_x, sizeof(uint32_t));
  cudaMalloc(&res_y, sizeof(uint32_t));
  reduce_mat<<<dim3((image_dimx + 15) / 16, (image_dimy + 15) / 16),
               dim3(16, 16)>>>(d_match_matrix, image_dimx, image_dimy);
  uint32_t x, y;
  cudaMemcpy(&x, res_x, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&y, res_y, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  // printf("x: %d, y: %d\n", x, y);

      return 0;
}

// WARNING: If you create your own bitmap, and then a cropped version of it
// make sure to do it in the right software, or you'll get the wrong results.
