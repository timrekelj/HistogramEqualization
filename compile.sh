#!/bin/bash

nvcc -o out/main main.cu
nvcc -o out/main-sequential main-sequential.cu
gcc -o out/main-sequential-c main-sequential.c -lm