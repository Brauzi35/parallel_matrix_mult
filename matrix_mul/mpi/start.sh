#! /bin/bash

rm -d -r target;


mkdir target


module load gnu mpich cuda;

mpic++ -lm -lstdc++  matrix_mul_transpose.c -o target/tras 

cd target
touch output.csv
touch errors.csv
echo "n_righeA,n_colonneA,n_righeB,n_colonneB,righe_blocco,colonne_blocco,n_proc,time,flops" > output.csv
echo "n_righeA,n_colonneA,n_righeB,n_colonneB,righe_blocco,colonne_blocco,n_proc,time,flops" > errors.csv



for dim in {32,64,128,256}
do 
    for block_dim in {1,2,3,4}
    do
        for proc in {1,2,3,4,8,12,16}
        do
            mpirun -n $proc ./tras $dim $dim $dim $dim $block_dim $block_dim
        done
    done
done

for dim in {1024,2048,4096,8192}
do 
    for block_dim in {2,3,4}
    do
        for proc in {2,8,16}
        do
            mpirun -n $proc ./tras $dim $dim $dim $dim $block_dim $block_dim
        done
    done
done

# Test per matrici rettangolari
for rowsA in {32,64,128,256}
do
    for colsA_rowsB in {32,64,128,256}  # ColsA deve essere uguale a RowsB
    do
        for colsB in {32,64,128,256}
        do
            for block_dim in {1,2,3,4}
            do
                for proc in {1,4,8,12,16}
                do
                    mpirun -n $proc ./tras $rowsA $colsA_rowsB $colsA_rowsB $colsB $block_dim $block_dim
                done
            done
        done
    done
done

for rowsA in {1024,2048}
do
    for colsA_rowsB in {512,1024}  # ColsA deve essere uguale a RowsB
    do
        for colsB in {1024,2048}
        do
            for block_dim in {2,3,4}
            do
                for proc in {2,8,16}
                do
                    mpirun -n $proc ./tras $rowsA $colsA_rowsB $colsA_rowsB $colsB $block_dim $block_dim
                done
            done
        done
    done
done

