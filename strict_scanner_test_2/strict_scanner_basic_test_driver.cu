#include<stdio.h>

#include "libstrictscanner.h"

int test1() {
    uint8_t image[4][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    uint8_t token[2][2] = {
        {6, 7},
        {10, 11}
    };

    bool expected_match_matrix[4][4] = {
        {false, false, false, false},
        {false, true, false, false},
        {false, false, false, false},
        {false, false, false, false}
    };

    bool returned_match_matrix[4][4] = {false};

    int result = strict_scan(
        (uint8_t *)image, 4, 4,
        (uint8_t *)token, 2, 2,
        1.0, (bool *)returned_match_matrix
    );

    if (result) {
        printf("strict_scan encountered a fatal error\n");
        return 1;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (expected_match_matrix[i][j] != returned_match_matrix[i][j]) {
                printf("Test 1 failed at %d, %d\n", i, j);
                return 1;
            }
        }
    }

    return 0;
}

int test2() {
    uint8_t image[1][1] = {
        {55}
    };

    uint8_t token[1][1] = {
        {44}
    };

    bool expected_match_matrix[1][1] = {
        {false}
    };

    bool returned_match_matrix[1][1] = {false};

    int result = strict_scan(
        (uint8_t *)image, 1, 1,
        (uint8_t *)token, 1, 1,
        1.0, (bool *)returned_match_matrix
    );

    if (result) {
        printf("strict_scan encountered a fatal error\n");
        return 1;
    }

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            if (expected_match_matrix[i][j] != returned_match_matrix[i][j]) {
                printf("Test 2 failed at %d, %d\n", i, j);
                return 1;
            }
        }
    }

    return 0;
}

int test3() {
    uint8_t image[1][1] = {
        {255}
    };

    uint8_t token[1][1] = {
        {255}
    };

    bool expected_match_matrix[1][1] = {
        {true}
    };

    bool returned_match_matrix[1][1] = {false};

    int result = strict_scan(
        (uint8_t *)image, 1, 1,
        (uint8_t *)token, 1, 1,
        1.0, (bool *)returned_match_matrix
    );

    if (result) {
        printf("strict_scan encountered a fatal error\n");
        return 1;
    }

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            if (expected_match_matrix[i][j] != returned_match_matrix[i][j]) {
                printf("Test 3 failed at %d, %d\n", i, j);
                return 1;
            }
        }
    }

    return 0;
}


int test4() {
    uint8_t image[3][4] = {
        {1, 2, 3, 4},
        {1, 2, 3, 5},
        {1, 2, 3, 4},
    };

    uint8_t token[1][3] = {
        {2, 3, 4}
    };

    bool expected_match_matrix[3][4] = {
        {false, true, false, false},
        {false, false, false, false},
        {false, true, false, false}
    };

    bool returned_match_matrix[3][4] = {false};

    int result = strict_scan(
        (uint8_t *)image, 3, 4,
        (uint8_t *)token, 1, 3,
        1.0, (bool *)returned_match_matrix
    );

    if (result) {
        printf("strict_scan encountered a fatal error\n");
        return 1;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            if (expected_match_matrix[i][j] != returned_match_matrix[i][j]) {
                printf("Test 4 failed at %d, %d\n", i, j);
                return 1;
            }
        }
    }

    return 0;
}

int main() {
    bool fail = 0;
    printf("Running tests:\n");

    int result1 = test1();
    printf("Test 1: ");
    if (result1) {
        printf("Failed\n");
        fail = 1;
    } else {
        printf("Passed\n");
    }

    int result2 = test2();
    printf("Test 2: ");
    if (result2) {
        printf("Failed\n");
        fail = 1;
    } else {
        printf("Passed\n");
    }

    int result3 = test3();
    printf("Test 3: ");
    if (result3) {
        printf("Failed\n");
        fail = 1;
    } else {
        printf("Passed\n");
    }

    int result4 = test4();
    printf("Test 4: ");
    if (result4) {
        printf("Failed\n");
        fail = 1;
    } else {
        printf("Passed\n");
    }

    printf("Tests complete\n");
    return fail;
}

// I am aware that this file is really badly written, but it's simple to copy-paste and test. LOL!
