#!/bin/bash
module load python/3.4.1-fasrc01
module load legacy/0.0.1-fasrc01
module load centos6/cython-0.20_python-3.3.2
module load gcc/6.2.0-fasrc02

srun --mem-per-cpu=4000 -p general -n 64 --pty -t 0-06:00 /bin/bash
python setup.py build_ext --inplace
