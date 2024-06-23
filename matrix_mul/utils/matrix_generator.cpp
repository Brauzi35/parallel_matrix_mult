#include "matrix_generator.h"
#include "matrix_utils.h"
#include <random>
#include <iostream>
#include <fstream>

#include <random>


using namespace std;
//typedef std::mt19937 rng_type; //mersenneTwister
//std::uniform_int_distribution<rng_type::result_type> udist(0.0f, 100.0f); //uniform distribution between [0.0, 100.0]

//rng_type rng;
float **alloc_arr_float(int rows,int cols){
    int i;
    float **array = (float **)malloc(rows * sizeof(float *));
    if (array == NULL) {
        fprintf(stderr, "Errore: impossibile allocare memoria per l'array di puntatori\n");
        exit(EXIT_FAILURE);
    }

    float *data = (float *)calloc(rows * cols, sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Errore: impossibile allocare memoria per i dati della matrice\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < rows; i++) {
        array[i] = &(data[cols * i]);
    }

    return array;
}

float **mat_gen(int rows, int cols, int seedval){
	
  float **matrix = alloc_arr_float(rows, cols);	
  //rng_type::result_type const seedval = 42;
  std::mt19937 rng(seedval);
  std::uniform_real_distribution<float> udist(0.0f, 100.0f); 
  
  //rng.seed(seedval); //planting seed
  
  
  
  for (int i = 0; i < rows; ++i) {
	  for(int j = 0; j<cols; j++){
		  matrix[i][j] = udist(rng);
	  }
            
            //printf("%f\n", matrix[i]);
  }
  
  std::cout << "Matrix successfully generated" << std::endl;
  
  return matrix;
}

int* mat_gen_int(int rows, int cols, int seedval) {
    int *matrix = (int *)malloc(rows * cols * sizeof(int));
    std::mt19937 rng(seedval);
    std::uniform_int_distribution<int> udist(0, 100); 
  
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = udist(rng);
    }
  
    std::cout << "Matrix successfully generated" << std::endl;
  
    return matrix;
}

