#!/bin/bash
#SBATCH -J Parallel_sum_%A
#SBATCH -o p_sum_%A_%a.out
#SBATCH -e p_sum_%A_%a.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=10000

## Load software ##
module load python/3.4.1-fasrc01
module load legacy/0.0.1-fasrc01
module load centos6/cython-0.20_python-3.3.2
module load gcc/6.2.0-fasrc02
module load hpc/pandas-0.11.0_python-2.7.1

## EXECUTE CODE ##
python setup.py build_ext --inplace

python main.py
