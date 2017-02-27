#!/bin/bash
module load python/3.4.1-fasrc01
module load legacy/0.0.1-fasrc01
module load centos6/cython-0.20_python-3.3.2
module load gcc/6.2.0-fasrc02
module load cuda/7.5-fasrc01

#srun --mem-per-cpu=40000 -p general -n 64 --pty -t 0-06:00 /bin/bash
#srun --mem-per-cpu=4000 -p holyseasgpu -n 64 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-06:00 /bin/bash

python setup.py build_ext --inplace

python main.py
