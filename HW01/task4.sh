#!/usr/bin/env bash
#SBATCH -J FirstSlurm
#SBATCH -p wacc
#SBATCH -t 0-00:02:30
#SBATCH -o FirstSlurm-%j.out -e FirstSlurm-%j.err
#SBATCH -c 2

echo $HOSTNAME
