#!/usr/bin/env bash
#SBATCH -J task1
#SBATCH -p wacc
#SBATCH -o task1-%j.out -e task1-%j.err

for i in {10..30}; do
	echo "n = 2^$i"
	./task1 $((2**i))
	echo
done
