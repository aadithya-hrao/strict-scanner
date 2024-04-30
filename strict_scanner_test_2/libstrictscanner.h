#pragma once

#include <stdint.h>
#include <stdbool.h>

int strict_scan(
    uint8_t* image, uint32_t image_dimx, uint32_t image_dimy,
    uint8_t* token, uint32_t token_dimx, uint32_t token_dimy, 
    float match_similarity_threshold, bool* match_matrix
);
