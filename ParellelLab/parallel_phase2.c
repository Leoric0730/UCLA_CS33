#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"

void parallel_to_grayscale(long img[DIM_ROW][DIM_COL][DIM_RGB], long ***grayscale_img, long *min_max_gray) {
    int row, col, pixel, gray_pixel;
    int min_gray = 256;
    int max_gray = -1;

    #pragma omp parallel for private(row, gray_pixel, pixel) reduction(min:min_gray) reduction(max:max_gray) collapse(2)
    for (row = 0; row < DIM_ROW; row++) {
        for (col = 0; col < DIM_COL; col++){
            // Calculate the sum of pixel values for each gray_pixel in parallel
            for (gray_pixel = 0; gray_pixel < DIM_RGB; gray_pixel++) {
                long gray_pixel_sum = 0;
                    gray_pixel_sum += img[row][col][0]+img[row][col][1]+img[row][col][2];
                
                // Calculate the average and store it in grayscale_img
                long gray_pixel_value = gray_pixel_sum / DIM_RGB;
                grayscale_img[row][col][gray_pixel] = gray_pixel_value;

                // Update min_gray and max_gray using the reduction clause
                min_gray = (gray_pixel_value < min_gray) ? gray_pixel_value : min_gray;
                max_gray = (gray_pixel_value > max_gray) ? gray_pixel_value : max_gray;
            }
        }
    }

    // Update min_max_gray after the parallel region
    #pragma omp single
    {
        min_max_gray[0] = min_gray;
        min_max_gray[1] = max_gray;
    }
}
