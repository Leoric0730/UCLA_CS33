#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"

//This code is NOT buggy, just sequential. Speed it up. 
void parallel_convolution(long img[DIM_ROW+PAD][DIM_COL+PAD][DIM_RGB], long kernel[DIM_KERNEL][DIM_KERNEL], long ***convolved_img) {
    int row, col, pixel, kernel_row, kernel_col;

    // Set number of threads for parallel execution
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    #pragma omp parallel for private(pixel, col, kernel_row, kernel_col) shared(img, kernel, convolved_img)
    for (row = 0; row < DIM_ROW; row++) {
        for (col = 0; col < DIM_COL; col++) {
            for (pixel = 0; pixel < DIM_RGB; pixel++) {
                for (kernel_col = 0; kernel_col < DIM_KERNEL; kernel_col++) {
                    for (kernel_row = 0; kernel_row < DIM_KERNEL; kernel_row++) {
                        convolved_img[row][col][pixel] += img[row + kernel_row][col + kernel_col][pixel] * kernel[kernel_row][kernel_col];
                    }
                }
                convolved_img[row][col][pixel] /= GBLUR_NORM;
            }
        }
    }
}
