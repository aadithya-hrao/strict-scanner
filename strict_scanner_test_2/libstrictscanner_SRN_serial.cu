// #include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "libstrictscanner.h"

// Strict scan that compares the token with the image pixel by pixel
// WARNING: Note that X*Y is basically an M*N matrix, where 
// M (X) is the height/vertical and N (Y) is the width/horizontal
int strict_scan(
    uint8_t* image, uint32_t image_dimx, uint32_t image_dimy,
    uint8_t* token, uint32_t token_dimx, uint32_t token_dimy, 
    float match_similarity_threshold, bool* match_matrix
) {
    // accuracy increases by multiplying here (instead of dividing the final value later)
    float error_threshold = (1 - match_similarity_threshold) * 255 * token_dimy * token_dimx;
    // printf("Error threshold: %f\n", error_threshold);

    for (uint32_t x = 0; x < image_dimx - token_dimx + 1; x++) {
        for (uint32_t y = 0; y < image_dimy - token_dimy + 1; y++) {
            float match_difference_error = 0;
            for (uint32_t tx = 0; tx < token_dimx; tx++) {
                for (uint32_t ty = 0; ty < token_dimy; ty++) {
                    match_difference_error += abs(image[(x + tx) * image_dimy + (y + ty)] - token[tx * token_dimy + ty]);
                }
            }
            if (match_difference_error <= error_threshold) {
                match_matrix[x * image_dimy + y] = true;
            }
        }
    }

    return 0;
}

// Yes, this file seems to HAVE to have a CUDA extension as I'm linking it to another CUDA file,
// even though this file doesn't have any CUDA code in it.
