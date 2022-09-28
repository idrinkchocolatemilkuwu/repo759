#!/usr/bin/env bash
#SBATCH -J task3
#SBATCH -p wacc
#SBATCH -t 0-00:30:00
#SBATCH -o task3-%j.out -e task3-%j.err
#SBATCH --gres=gpu:1

module load nvidia/cuda
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

for i in {10..29}; do
	./task3 $((2**i))
	echo
done
