#include<stdio.h>

#include "libstrictscanner.h"

void random_fill(uint8_t* image, int image_dimx, int image_dimy) {
    for (int i = 0; i < image_dimx; i++) {
        for (int j = 0; j < image_dimy; j++) {
            image[i * image_dimy + j] = rand() % 256;
        }
    }
}

int token_crop(uint8_t* image, int image_dimx, int image_dimy, uint8_t* token, int token_dimx, int token_dimy, int token_offsetx, int token_offsety) {
    if (token_offsetx + token_dimx > image_dimx || token_offsety + token_dimy > image_dimy) {
        return 1;
    }

    for (int i = 0; i < token_dimx; i++) {
        for (int j = 0; j < token_dimy; j++) {
            token[i * token_dimy + j] = image[(token_offsetx + i) * image_dimy + (token_offsety + j)];
        }
    }

    return 0;
}

int run() {
    const int image_dimx = 10000;
    const int image_dimy = 9000;

    const int token_offsetx = 1;
    const int token_offsety = 2;

    const int token_dimx = 9999;
    const int token_dimy = 8997;

    // const int image_dimx = 4;
    // const int image_dimy = 4;

    // const int token_offsetx = 1;
    // const int token_offsety = 2;

    // const int token_dimx = 2;
    // const int token_dimy = 2;

    const float match_similarity_threshold = 1.0;

    //In this test, I'm generating at runtime, but finally I'll likely do a static testing
    uint8_t* image = (uint8_t*)malloc(image_dimx * image_dimy * sizeof(uint8_t));
    random_fill(image, image_dimx, image_dimy);
    printf("Image generated\n");

    uint8_t* token = (uint8_t*)malloc(token_dimx * token_dimy * sizeof(uint8_t));
    int crop_result = token_crop(image, image_dimx, image_dimy, token, token_dimx, token_dimy, token_offsetx, token_offsety);
    if (crop_result) {
        printf("Token crop failed\n");
        return 1;
    }
    printf("Token cropped\n");

    bool* match_matrix = (bool*)malloc(image_dimx * image_dimy * sizeof(bool));
    for (int i = 0; i < image_dimx; i++) {
        for (int j = 0; j < image_dimy; j++) {
            match_matrix[i * image_dimy + j] = false;
        }
    }

    int result = strict_scan(
        image, image_dimx, image_dimy,
        token, token_dimx, token_dimy,
        match_similarity_threshold, match_matrix
    );

    if (result) {
        printf("strict_scan encountered a fatal error\n");
        return 1;
    }

    if (match_matrix[token_offsetx * image_dimy + token_offsety] == false) {
        printf("no match found at expected[%d, %d]\n", token_offsetx, token_offsety);
        return 1;
    }
    
    return 0;
}


int main() {
    bool fail = 0;
    printf("Running runs:\n");

    int result1 = run();
    printf("Run 1: ");
    if (result1) {
        printf("Failed\n");
        fail = 1;
    } else {
        printf("Passed\n");
    }

    printf("runs complete\n");
    return fail;
}

// I am aware that this file is really badly written, but it's simple to copy-paste and test. LOL!
