#!/bin/bash

## Load software ##
module load python/3.4.1-fasrc01
module load legacy/0.0.1-fasrc01
module load centos6/cython-0.20_python-3.3.2
module load gcc/6.2.0-fasrc01

## EXECUTE CODE ##
python setup.py build_ext --inplace
