#!/usr/bin/env bash
#SBATCH -J task1
#SBATCH -p wacc
#SBATCH -t 0-00:02:30
#SBATCH -o task1-%j.out -e task1-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda
nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
./task1
