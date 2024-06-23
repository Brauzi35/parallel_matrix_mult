#! /bin/bash

rm -d -r target;

mkdir target

module load gnu mpich cuda;

nvcc mat_mul_cuda.cu -o cuda  

cd target
touch output.csv

echo "n_righeA,n_colonneA,n_righeB,n_colonneB,time,flops" > output.csv

cd ..

for dim in {32,64,128,256,512,1024,2048,4096,8192} //squared
do
    ./cuda $dim $dim $dim

done

for m in {32,64,128,256,512,1024,2048,4096,8192} //rect
do
    for n in {32,64,128,256,512,1024,2048,4096,8192}
    do

        for k in {32,64,128,256,512,1024,2048,4096,8192}
        do
            ./cuda $m $n $k

        done

    done

done