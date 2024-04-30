#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "libstrictscanner.h"

// THIS IS JUST AN EXAMPLE, YOU CAN CHANGE THE KERNEL TO WHATEVER YOU WANT
// YOU CAN ALSO HAVE AS MANY KERNELS AS YOU WANT, and name it whatever,
// and write whatever, I don't care. Just don't do rm -rf LOL!
__global__ void strict_scan_kernel(uint8_t *image, uint32_t image_dimx,
                                   uint32_t image_dimy, uint8_t *token,
                                   uint32_t token_dimx, uint32_t token_dimy,
                                   float error_threshold, bool *match_matrix) {

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

// Strict scan that compares the token with the image pixel by pixel
// WARNING: Note that X*Y is basically an M*N matrix, where
// M (X) is the height/vertical and N (Y) is the width/horizontal
//

// THIS FUNCTION SIGNATURE MUST NOT CHANGE!!! AS I'M GOING TO CALL THIS FROM
// OUTSIDE USING MY CUSTOM TEST AND PERF ANALYSIS FRAMEWORK
int strict_scan(uint8_t *image, uint32_t image_dimx, uint32_t image_dimy,
                uint8_t *token, uint32_t token_dimx, uint32_t token_dimy,
                float match_similarity_threshold, bool *match_matrix) {
  // DO WHAT YOU WANT HERE. YOU CAN WRITE SERIAL CODE, LAUNCH KERNELS,
  // WHATEVER...
  // Just don't do rm -rf / LOL!

  // refer the serialexample to see how to do the whole thing serially
  // it won't get you marks for writing, but you might get some marks during
  // perf eval, if your serial code is not too slow. lol!

  float error_threshold =
      (1 - match_similarity_threshold) * 255 * token_dimy * token_dimx;
  printf("Error threshold: %f\n", error_threshold);
  // example

    uint8_t *d_image, *d_token;
    bool *d_match;
  cudaMalloc(&d_image, (image_dimx * image_dimy));
  cudaMalloc(&d_token, (token_dimx * token_dimy));
  cudaMalloc(&d_match, (image_dimx * image_dimy));

  cudaMemcpy(d_image, image, image_dimx * image_dimy, cudaMemcpyHostToDevice);
  cudaMemcpy(d_token, token, token_dimx * token_dimy, cudaMemcpyHostToDevice);
  dim3 block(16, 16);
  dim3 grid((image_dimx + block.x - 1) / block.x,
            (image_dimy + block.y - 1) / block.y);
  strict_scan_kernel<<<grid, block>>>(d_image, image_dimx, image_dimy, d_token,
                                      token_dimx, token_dimy, error_threshold,
                                      d_match);
    cudaMemcpy(match_matrix, d_match, image_dimx * image_dimy,
             cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_token);
    cudaFree(d_match);

  return 0;
}

// YOU WILL BE SUBMITTING THIS FILE TO ME. NOTHING ELSE!
