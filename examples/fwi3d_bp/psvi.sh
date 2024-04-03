#!/bin/bash

#$ -S /bin/bash
#$ -j y
#$ -P eip
#$ -N psvi_lowf
# -V
# longest time
#$ -l d_rt=50:00:00
#$ -l lustre03=1
#$ -pe cascadelake 96
#$ -cwd
#$ -o psvi_lowfreq.log
# job array
# -t 1-1:1

module load sge
module load intel
# module load pyhpc
module load conda
conda activate intel-2019

export OMP_NUM_THREADS=36

PYTHONPATH=$(pwd) python fwi3d.py
