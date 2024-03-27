#!/bin/bash

# set number of threads used for openmp
export OMP_NUM_THREADS=36
python fwi3d.py 
