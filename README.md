# Histogram Equalization - Final Project Assignment

This project contains three different programs:
 - `main.cu` - parallel implementation in CUDA
 - `main-sequential.cu` - sequential implementation in CUDA
 - `main-sequential.c` - sequential implementation in C

## How to run

### Only one file
1. Compile
```
nvcc -o out/main main.cu
```

> if you want to compile C file, not CUDA:
> ```
> gcc -o out/main-sequential-c main-sequential.c -lm
> ```

2. Run (first argument is image name, second argument is how many times you want to run program)
```
srun --reservation=psistemi --partition=gpu --gpus=1 out/main images-input/kolesar-neq.jpg 1
```

--- 

### All three files
1. Compile
```
./compile.sh
```

2. Run (first argument is image name, second argument is how many times you want to run program)
```
./run.sh images-input/kolesar-neq.jpg 10
```