#include <stdlib.h>
#include <stdio.h>
#include "matrix_utils.h"



void write_matrix_to_file(const char *filename, float *matrix, int rows, int cols) {
    FILE *file;

    if ((file = fopen(filename, "wb")) == NULL) {
        perror("Error in opening file");
        exit(EXIT_FAILURE);
    }

    fwrite(matrix, sizeof(float), rows * cols, file);
    fclose(file);
}

float* read_matrix_from_file(const char *filename, int rows, int cols) {
    FILE *file;
    float *matrix;

    if ((file = fopen(filename, "rb")) == NULL) {
        perror("Error in opening file");
        exit(EXIT_FAILURE);
    }

    matrix = (float *)malloc((rows) * (cols) * sizeof(int));
    fread(matrix, sizeof(float), (rows) * (cols), file);
    fclose(file);

    return matrix;
}