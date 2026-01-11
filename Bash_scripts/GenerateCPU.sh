#!/bin/bash
#PBS -N CNNGPU
#PBS -l walltime=20:10:10
#PBS -l select=1:mem=64gb:ncpus=16:ngpus=1

cd $HOME/CNN/Old/ML/src/generation/ 

module load anaconda3/personal

python Generator_9mm.py 9mmOutputs 10000 500 9 1



