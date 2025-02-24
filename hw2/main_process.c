#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "png_util.h"

#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define max(X, Y) ((X) > (Y) ? (X) : (Y))

void abort_(const char *s, ...) {
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void apply_average_filter(char **img, char **output, image_size_t sz, int halfwindow) {
    #pragma omp parallel for schedule(static) // Parallelize rows
    for (int r = 0; r < sz.height; r++) {
        for (int c = 0; c < sz.width; c++) {
            double count = 0;
            double tot = 0;
            for (int rw = max(0, r - halfwindow); rw < min(sz.height, r + halfwindow + 1); rw++) {
                for (int cw = max(0, c - halfwindow); cw < min(sz.width, c + halfwindow + 1); cw++) {
                    count++;
                    tot += (double)img[rw][cw];
                }
            }
            output[r][c] = (int)(tot / count);
        }
    }
}

void apply_sobel_filter(char **img, double **gradient, image_size_t sz) {
    double xfilter[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    double yfilter[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    #pragma omp parallel for schedule(static) // Parallelize rows
    for (int r = 1; r < sz.height - 1; r++) {
        for (int c = 1; c < sz.width - 1; c++) {
            double Gx = 0;
            double Gy = 0;
            for (int rw = 0; rw < 3; rw++) {
                for (int cw = 0; cw < 3; cw++) {
                    Gx += ((double)img[r + rw - 1][c + cw - 1]) * xfilter[rw][cw];
                    Gy += ((double)img[r + rw - 1][c + cw - 1]) * yfilter[rw][cw];
                }
            }
            gradient[r][c] = sqrt(Gx * Gx + Gy * Gy);
        }
    }
}

void apply_threshold(double **gradient, char **output, image_size_t sz, double thresh) {
    #pragma omp parallel for schedule(static) // Parallelize rows
    for (int r = 0; r < sz.height; r++) {
        for (int c = 0; c < sz.width; c++) {
            output[r][c] = (gradient[r][c] > thresh) ? 255 : 0;
        }
    }
}

char **process_img(char **img, char **output, image_size_t sz, int halfwindow, double thresh) {
    clock_t start_total = clock();

    // Apply average filter
    clock_t start_avg = clock();
    apply_average_filter(img, output, sz, halfwindow);
    clock_t end_avg = clock();
    printf("Average Filter Execution Time: %.4f seconds\n", (double)(end_avg - start_avg) / CLOCKS_PER_SEC);

    // Apply Sobel filter and calculate gradient
    clock_t start_gradient = clock();
    double *gradient = (double *)malloc(sz.width * sz.height * sizeof(double));
    double **g_img = malloc(sz.height * sizeof(double *));
    for (int r = 0; r < sz.height; r++) {
        g_img[r] = &gradient[r * sz.width];
    }
    apply_sobel_filter(output, g_img, sz);
    clock_t end_gradient = clock();
    printf("Gradient Calculation Execution Time: %.4f seconds\n", (double)(end_gradient - start_gradient) / CLOCKS_PER_SEC);

    // Apply thresholding
    clock_t start_thresh = clock();
    apply_threshold(g_img, output, sz, thresh);
    clock_t end_thresh = clock();
    printf("Thresholding Execution Time: %.4f seconds\n", (double)(end_thresh - start_thresh) / CLOCKS_PER_SEC);

    // Free allocated memory
    free(gradient);
    free(g_img);

    clock_t end_total = clock();
    printf("Total Execution Time: %.4f seconds\n", (double)(end_total - start_total) / CLOCKS_PER_SEC);

    return output;
}

int main(int argc, char **argv) {
    int channels = 1;
    double thresh = 50;
    int halfwindow = 3;

    if (argc < 3) {
        abort_("Usage: process <file_in> <file_out> <halfwindow=3> <threshold=50>");
    }

    if (argc > 3) {
        halfwindow = atoi(argv[3]);
    }

    if (argc > 4) {
        thresh = (double)atoi(argv[4]);
    }

    image_size_t sz = get_image_size(argv[1]);
    char *s_img = (char *)malloc(sz.width * sz.height * channels * sizeof(char));
    char *o_img = (char *)malloc(sz.width * sz.height * channels * sizeof(char));

    read_png_file(argv[1], (unsigned char *)s_img, sz);

    char **img = malloc(sz.height * sizeof(char *));
    for (int r = 0; r < sz.height; r++) {
        img[r] = &s_img[r * sz.width];
    }
    char **output = malloc(sz.height * sizeof(char *));
    for (int r = 0; r < sz.height; r++) {
        output[r] = &o_img[r * sz.width];
    }

    process_img(img, output, sz, halfwindow, thresh);

    write_png_file(argv[2], (unsigned char *)o_img, sz);

    free(s_img);
    free(o_img);
    free(img);
    free(output);

    return 0;
}
