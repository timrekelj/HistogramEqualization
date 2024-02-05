#!/bin/bash

# example of execution: ./run.sh images-input/kolesar-neq.jpg 10
# first argument is input image, second is how many times do we want to run each program

echo "--------- IMAGE DATA ---------"
echo $(file $1)

echo ""
echo "--------- RUNNING PARALLEL ---------"
srun --reservation=psistemi --partition=gpu --gpus=1 out/main $1 $2

echo ""
echo "--------- RUNNING SEQUENTIAL ---------"
srun --reservation=psistemi --partition=gpu --gpus=1 out/main-sequential $1 $2

echo ""
echo "--------- RUNNING SEQUENTIAL WITHOUT CUDA ---------"
srun --reservation=psistemi --partition=gpu --gpus=1 out/main-sequential-c $1 $2