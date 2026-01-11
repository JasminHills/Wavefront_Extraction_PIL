#!/bin/bash
#PBS -N CNNGPU
#PBS -l walltime=48:10:10
#PBS -l select=1:mem=64gb:ncpus=16:ngpus=1

cd $HOME/CNN/Old/ML/src/generation/ # change to the directory where the code is located
module load anaconda3/personal

python Train_PIL_9mm.py
