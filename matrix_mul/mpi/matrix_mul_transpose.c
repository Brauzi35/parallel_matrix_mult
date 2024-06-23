#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"
#include "/home/vbrauzi/matrix_mul/utils/matrix_generator.cpp"


/*
#define ROW_A 1028
#define ROW_B 156
#define COL_A 156
#define COL_B 1028
#define BLOCk_COL 2
#define BLOCk_ROW 2
*/

float **alloc_2d_float(int rows,int cols){
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

int **alloc_2d_int(int rows,int cols){
    int i ;
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array = (int **)malloc(rows*sizeof(int*));
    for (i=0; i<rows; i++) {
        array[i] = &(data[cols*i]);
    }
    return array;
}

int array_dim_calc(int mat_rows, int mat_cols, int block_rows, int block_cols, int rank){

	//capire a che riga e colonna sto all'inizio
	int curr_row = floor((double)rank/block_rows);
	int curr_col = rank%block_cols;
	
	int row_spot = 0;
	int col_spot = 0;
	
	for(int i = curr_col; i<mat_cols; i += block_cols){
		row_spot++;
	}
	for(int j = curr_row; j<mat_rows; j += block_rows){
		col_spot++;
	}
	
	int dim_array = col_spot*row_spot; //dimensione massima array delle coordinate
	printf("rank %d parte dalle coordinate %d,%d: row_spot = %d, col_spot = %d\n", rank, curr_row, curr_col, row_spot, col_spot);
	return dim_array;
	
	
	
}

int **mat_mapper(int mat_rows, int mat_cols, int block_rows, int block_cols, int rank, int procs){

    int block_dim = block_cols*block_rows;
    int count = 0;
    //phase 1: build array representing which block_element is managed by process i
    /*eg 
        |x|x|x|
        |x|x|x| --> with 4 processes [0,1,2,3,0,1]
    */

    int *block_array = (int *)malloc(block_dim*sizeof(int));
    for(int k = 0; k<block_dim; k++){
        block_array[k] = k%procs;
    }
    //build block toponomy

    int **block = alloc_2d_int(block_rows, block_cols);
    for(int q = 0; q<block_rows; q++){
        for(int s = 0; s<block_cols; s++){
            block[q][s] = (q + 1)*s;
        }

    }
    //phase 2: map matrix elements in block elements
    int **mapped_matrix = alloc_2d_int(mat_rows, mat_cols);
    for(int i = 0; i<mat_rows; i++){
        int x = i % block_rows;
        for(int j = 0; j<mat_cols; j++){
            
            int y = j % block_cols;
            mapped_matrix[i][j] = block[x][y]; //si può togliere
            // map mapped_matrix elements into block array
            mapped_matrix[i][j] = block_array[mapped_matrix[i][j]];

        }

    }
    //phase 3: 

    return mapped_matrix;


}

int array_dim_calc_lessProcs(int **mapped_matrix, int rank, int mat_rows, int mat_cols){
    int count = 0;
    for(int i = 0; i<mat_rows; i++){
        for(int j = 0; j<mat_cols; j++){
            if(mapped_matrix[i][j]==rank){
                count++;
            }

        }
    }
    return count;
}


int **blocks_calc(int *matrix, int mat_rows, int mat_cols, int block_rows, int block_cols, int dim_array,int rank){
	
	//capire a che riga e colonna sto all'inizio
	int curr_row = floor((double)rank/block_rows);
	int curr_col = rank%block_cols;
	double rows_div = (double)mat_rows/block_rows;
	
	
	int **array = alloc_2d_int(dim_array, 2);

	int count = 0;
	//voglio capire quali elementi della matrice gestisce ogni singolo processo
	for(int i = curr_row; i<mat_rows; i+= block_rows){
		for(int j = curr_col; j<mat_cols; j+= block_cols){
			array[count][0] = i;
			array[count][1] = j;
			count+=1;
			//printf("il processo con rank %d sarà responsabile della coppia %d,%d\n",rank ,i, j);
			
		}
	}
	
	return array;
	
}



int **blocks_calc2(int **mapped_matrix, int mat_rows, int mat_cols, int rank, int dim_array){
    int **array = alloc_2d_int(dim_array, 2);
    int idx = 0;
    for(int i = 0; i<mat_rows; i++){
        for(int j= 0; j<mat_cols; j++){
            if(mapped_matrix[i][j]==rank){
                array[idx][0] = i;
                array[idx][1] = j;
                idx++;
            }
        }
    }
    return array;

}

float *prod(float **matrix, float **matrixB,int **coord, int dim_array, int mat_rows, int mat_cols, int matB_rows, int matB_cols, int my_rank){

	float *ret = (float *)calloc(dim_array, sizeof(float));
    if (ret == NULL) {
        fprintf(stderr, "Errore: impossibile allocare memoria per il risultato\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dim_array; i++) {
        int row = coord[i][0];
        int col = coord[i][1];
        
        // Calcola il prodotto scalare tra la riga di matrix e la colonna di matrixB
        float res = 0;
        for (int k = 0; k < mat_cols; k++) {
            res += matrix[row][k] * matrixB[col][k]; //invertito per trasposta
        }
        
        ret[i] = res;
    }
    
    return ret;

}

//builds a temp matrix to create A*B, putting the right value in the right coordinae x,y. Then this matrix will be summed w the others
float **temp_Cmatrix_builder(int **coord, float *prod, int mat_rows, int mat_cols, int dim_array){
	
	float **ret_mat = alloc_2d_float(mat_rows, mat_cols); //forse devo inizializazre a 0 tutti gli elementi prima DOVEVO INFATTI FUNZIONA FINO A QUI
	/*
    for (int i = 0; i < mat_rows; i++) {
        for (int j = 0; j < mat_cols; j++) {
            ret_mat[i][j] = 0;
        }
	}
    */
    
    //printf("ret_mat ok\n");
    
    for (int j = 0; j<dim_array; j++) {
		if(j>0 && coord[j][0] == 0 && coord[j][1]== 0){
			break;
		}
		
		//printf("prod = %d, x,y = %d,%d\n", prod[j], coord[j][0], coord[j][1]);
		ret_mat[coord[j][0]][coord[j][1]] += prod[j];
		
	}
	
	
    
    return ret_mat;
    
	
}

// Funzione per sommare due matrici
float **sommaMatrici(float **matrice1, float **matrice2, int mat_rows, int mat_cols) {
	float **risultato = alloc_2d_float(mat_rows, mat_cols);
    for (int i = 0; i < mat_rows; i++) {
        for (int j = 0; j < mat_cols; j++) {
			//printf("iterazione j=%d\n", j);
            risultato[i][j] = matrice1[i][j] + matrice2[i][j];
        }
    }
    return risultato;
}

bool areMatricesEqual(int rows, int cols, float** mat1, float** mat2) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mat1[i][j] != mat2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

// mat prod for check 
float** multiplyMatrices(int rowsA, int colsA, int colsB, float** matA, float** matB) {
    float** result = alloc_2d_float(rowsA, colsB);
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i][j] = 0;
            for (int k = 0; k < colsA; k++) {
                result[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return result;
}


int main(int argc, char** argv) {

    //init mpi
    

    int ROW_A = atoi(argv[1]);//1028;
    int ROW_B = atoi(argv[3]);//156;
    int COL_A = atoi(argv[2]);//156;
    int COL_B = atoi(argv[4]);//1028;
    int BLOCk_COL = atoi(argv[5]);//2;
    int BLOCk_ROW = atoi(argv[6]);//2;
    
    
    int tot_proc, my_rank;
    MPI_Comm comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    double start, end;
    start = MPI_Wtime();

    MPI_Comm_size(comm, &tot_proc);
    MPI_Comm_rank(comm, &my_rank);

    if(tot_proc>BLOCk_COL*BLOCk_ROW){
        if(my_rank == 0){
            printf("cannon have more procs than grid cells, exiting\n");
        }
        MPI_Abort(comm, EXIT_FAILURE);
        return 0;
    }
    if(argc > 7){

        if(my_rank == 0){
            printf("correct usage: mpirun -n $proc ./tras $rowsA $colsA_rowsB $colsA_rowsB $colsB $block_dim $block_dim\n");
        }
        MPI_Abort(comm, EXIT_FAILURE);
        return 0;

    }
    if(ROW_A<BLOCk_ROW || COL_B<BLOCk_COL){
        if(my_rank == 0){
            printf("matrix grid should be smaller then the matrix itself in both rows and columns\n");
        }
        MPI_Abort(comm, EXIT_FAILURE);
        return 0;

    }


    float **matrix = alloc_2d_float(ROW_A,COL_A);//mat_gen(ROW_A, COL_A, 42);	alloc_2d_float(ROW_A,COL_A);
    float **matrixB = alloc_2d_float(ROW_B,COL_B);//mat_gen(ROW_B, COL_B, 42+1);	//alloc_2d_float(ROW_B,COL_B);

    if(my_rank==0){
        matrix = mat_gen(ROW_A, COL_A, 42);	//alloc_2d_float(ROW_A,COL_A);
        matrixB = mat_gen(ROW_B, COL_B, 42+1);	//alloc_2d_float(ROW_B,COL_B);
        //MPI_Send(*matrix, ROW_A*COL_A, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        
        
    }
    MPI_Bcast( *matrix, ROW_A*COL_A, MPI_FLOAT, 0, MPI_COMM_WORLD );
    MPI_Bcast( *matrixB, ROW_B*COL_B, MPI_FLOAT, 0, MPI_COMM_WORLD );
    

    //for memory access reasons let's transpose matrixB

    float **matrixB_trps = alloc_2d_float(COL_B,ROW_B);

    for (int i = 0; i < COL_B; i++) {
        for (int j = 0; j < ROW_B; j++) {
            
            matrixB_trps[i][j] = matrixB[j][i];
        }
    }
    
    
    
    int dim_array;
    //printf("dimensione array: %d, myrank = %d\n", dim_array, my_rank);
    int **coord; //my_rank 
    //printf("coordinate trovate, myrank = %d\n", my_rank);
    if(tot_proc<BLOCk_COL*BLOCk_ROW){ //se ho meno processi che elementi nel blocco
        //dim_array = array_dim_calc_lessProcs(ROW_A,COL_B,BLOCk_ROW,BLOCk_COL, my_rank, tot_proc);
        int **mapped_matrix = mat_mapper(ROW_A,COL_B,BLOCk_ROW,BLOCk_COL, my_rank, tot_proc);
        dim_array = array_dim_calc_lessProcs(mapped_matrix, my_rank, ROW_A, COL_B);
        //coord = blocks_calc2((int *)matrix, ROW_A,COL_B,BLOCk_ROW,BLOCk_COL,dim_array,my_rank, tot_proc); //my_rank 
        coord = blocks_calc2(mapped_matrix, ROW_A,COL_B,my_rank,dim_array);
        free(mapped_matrix);

    }else{
        dim_array = array_dim_calc(ROW_A,COL_B,BLOCk_ROW,BLOCk_COL, my_rank);
        coord = blocks_calc((int *)matrix, ROW_A,COL_B,BLOCk_ROW,BLOCk_COL,dim_array,my_rank); //my_rank 

    }
    
    float *dotp = prod(matrix, matrixB_trps, coord, dim_array, ROW_A, COL_A, ROW_B, COL_B, my_rank);
    //printf("prodotto effettuato, myrank = %d\n", my_rank);
    
    float **temp_mat = temp_Cmatrix_builder(coord, dotp, ROW_A, COL_B,dim_array);
    //printf("ok fino a dopo temp_mat, my rank = %d\n", my_rank);
    float **mat_recived = alloc_2d_float(ROW_A, COL_B);
    
    
    
    
    
    // Raccolta delle matrici dai processi con rank > 0
    if (my_rank == 0) {
        float **mat_sum = alloc_2d_float(ROW_A, COL_B);
        // Processo 0 riceve e somma le matrici
        mat_sum = temp_mat;
        for (int i = 1; i < tot_proc; i++) {
            MPI_Recv(*mat_recived, ROW_A*COL_B, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("iterazione %d, primo valore matrice ricevuta = %f\n", i, mat_recived[1][0]);
            
            
            mat_sum = sommaMatrici(mat_sum, mat_recived, ROW_A, COL_B);
            //printf("iterazione %d ma fatta somma\n", i);
        }
        end = MPI_Wtime(); //here ends the program

        //verify if result is correct
        if(ROW_A < 2048 && COL_A < 2048 && COL_B < 2024){

            float **zero_prod = multiplyMatrices(ROW_A, COL_A, COL_B, matrix, matrixB); //int rowsA, int colsA, int colsB, float** matA, float** matB
            if(!areMatricesEqual(ROW_A,COL_B,zero_prod, mat_sum)){
                printf("error in prod, exiting\n");

                // Apertura del file di output in modalità scrittura
                FILE *output_file = fopen("errors.csv", "a");
                if (output_file == NULL) {
                fprintf(stderr, "Impossibile aprire il file di output.\n");
                return 1;
                }

                // Scrittura della stringa formattata nel file
                fprintf(output_file, "%d,%d,%d,%d,%d,%d,%d,%.6f,%.2f\n",
                ROW_A, COL_A, ROW_B, COL_B,
                BLOCk_ROW, BLOCk_COL, tot_proc, 0.0000, 0.0);

                return 0;
            }else{
                printf("the computation was correct!\n");
            }
            

        }

        
        
        
    } else {
        
        MPI_Send(*temp_mat, ROW_A*COL_B, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
    }
    
    
    float elapsed_time = end-start;
    


	
	if(my_rank==0){
        printf("time: %f\n", elapsed_time);
        double flops = (2.0*ROW_A*COL_A*COL_B)/elapsed_time;
        printf("flops: %f\n", flops);

        

        // Apertura del file di output in modalità scrittura
        FILE *output_file = fopen("output.csv", "a");
        if (output_file == NULL) {
           fprintf(stderr, "Impossibile aprire il file di output.\n");
           return 1;
        }

        
        // Scrittura della stringa formattata nel file
        fprintf(output_file, "%d,%d,%d,%d,%d,%d,%d,%.6f,%.2f\n",
            ROW_A, COL_A, ROW_B, COL_B,
            BLOCk_ROW, BLOCk_COL, tot_proc, elapsed_time, flops);

        // Chiusura del file
        fclose(output_file);

    }
    
    //printf("%d\n", tot_proc);
    free(matrix);
    free(matrixB);
    free(matrixB_trps);
    free(dotp);
    free(temp_mat);
    free(coord);
	
    MPI_Finalize();
    
    return 0;
	
}
