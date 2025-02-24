#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>  // Include for timing
#include "png_util.h"

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

void abort_(const char * s, ...)
{
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

char ** process_img(char ** img, char ** output, image_size_t sz, int halfwindow, double thresh)
{
    // Start timing for the entire process_img function
    clock_t start_total = clock();

    // Start timing for the Average Filter
    clock_t start_avg = clock();

    // Average Filter 
    for(int c=0;c<sz.width;c++) 
        for(int r=0;r<sz.height;r++)
        {
            double count = 0;
            double tot = 0;
            for(int cw=max(0,c-halfwindow); cw<min(sz.width,c+halfwindow+1); cw++)
                for(int rw=max(0,r-halfwindow); rw<min(sz.height,r+halfwindow+1); rw++)
                {
                    count++;
                    tot += (double) img[rw][cw];
                }
            output[r][c] = (int) (tot/count);
        }

    // End timing for the Average Filter
    clock_t end_avg = clock();
    double elapsed_avg = (double)(end_avg - start_avg) / CLOCKS_PER_SEC;
    printf("Average Filter Execution Time: %.4f seconds\n", elapsed_avg);

    //write debug image
    //write_png_file("after_smooth.png",output[0],sz);

    // Start timing for the Sobel Filters and Gradient Calculation
    clock_t start_gradient = clock();

    // Sobel Filters
    double xfilter[3][3];
    double yfilter[3][3];
    xfilter[0][0] = -1;
    xfilter[1][0] = -2;
    xfilter[2][0] = -1;
    xfilter[0][1] = 0;
    xfilter[1][1] = 0;
    xfilter[2][1] = 0;
    xfilter[0][2] = 1;
    xfilter[1][2] = 2;
    xfilter[2][2] = 1;
    for(int i=0;i<3;i++) 
        for(int j=0;j<3;j++)
            yfilter[j][i] = xfilter[i][j];

    double * gradient = (double *) malloc(sz.width*sz.height*sizeof(double));
    double ** g_img = malloc(sz.height * sizeof(double*));
    for (int r=0; r<sz.height; r++)
        g_img[r] = &gradient[r*sz.width];

    // Gradient filter
    for(int c=1;c<sz.width-1;c++)
        for(int r=1;r<sz.height-1;r++)
        {
            double Gx = 0;
            double Gy = 0;
            for(int cw=0; cw<3; cw++)
                for(int rw=0; rw<3; rw++)
                {
                    Gx +=  ((double) output[r+rw-1][c+cw-1])*xfilter[rw][cw];
                    Gy +=  ((double) output[r+rw-1][c+cw-1])*yfilter[rw][cw];
                }
            g_img[r][c] = sqrt(Gx*Gx+Gy*Gy);
        }

    // End timing for the Gradient Calculation
    clock_t end_gradient = clock();
    double elapsed_gradient = (double)(end_gradient - start_gradient) / CLOCKS_PER_SEC;
    printf("Gradient Calculation Execution Time: %.4f seconds\n", elapsed_gradient);

    // Start timing for Thresholding
    clock_t start_thresh = clock();

    // Thresholding
    for(int c=0;c<sz.width;c++)
        for(int r=0;r<sz.height;r++)
            if (g_img[r][c] > thresh)
                output[r][c] = 255;
            else
                output[r][c] = 0;

    // End timing for Thresholding
    clock_t end_thresh = clock();
    double elapsed_thresh = (double)(end_thresh - start_thresh) / CLOCKS_PER_SEC;
    printf("Thresholding Execution Time: %.4f seconds\n", elapsed_thresh);

    // End timing for the entire process_img function
    clock_t end_total = clock();
    double elapsed_total = (double)(end_total - start_total) / CLOCKS_PER_SEC;
    printf("Total Execution Time: %.4f seconds\n", elapsed_total);

    return output;
}

int main(int argc, char **argv)
{
    //Code currently does not support more than one channel (i.e. grayscale only)
    int channels=1; 
    double thresh = 50;
    int halfwindow = 3;

    //Ensure at least two input arguments
    if (argc < 3 )
        abort_("Usage: process <file_in> <file_out> <halfwindow=3> <threshold=50>");

    //Set optional window argument
    if (argc > 3 )
        halfwindow = atoi(argv[3]);

    //Set optional threshold argument
    if (argc > 4 )
        thresh = (double) atoi(argv[4]);

    //Allocate memory for images
    image_size_t sz = get_image_size(argv[1]);
    char * s_img = (char *) malloc(sz.width*sz.height*channels*sizeof(char));
    char * o_img = (char *) malloc(sz.width*sz.height*channels*sizeof(char));

    //Read in serial 1D memory
    read_png_file(argv[1],s_img,sz);

    //make 2D pointer arrays from 1D image arrays
    char **img = malloc(sz.height * sizeof(char*));
    for (int r=0; r<sz.height; r++)
        img[r] = &s_img[r*sz.width];
    char **output = malloc(sz.height * sizeof(char*));
    for (int r=0; r<sz.height; r++)
        output[r] = &o_img[r*sz.width];

    //Run the main image processing function
    process_img(img,output,sz,halfwindow,thresh);

    //Write out output image using 1D serial pointer
    write_png_file(argv[2],o_img,sz);

    return 0;
}
