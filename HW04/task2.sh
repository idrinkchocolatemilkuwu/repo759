#!/usr/bin/env bash
#SBATCH -J task2
#SBATCH -p wacc
#SBATCH -t 0-00:30:00
#SBATCH -o task2-%j.out -e task2-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

for i in {10..29}; do
	./task2 $((2**i)) 128 1024
	echo
done
