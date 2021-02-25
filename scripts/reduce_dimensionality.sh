#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=150GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=reducedim
#SBATCH --output=reducedim_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'adept' --model 'say'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'adept' --model 'spatial' --ndim 64 
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'adept' --model 'wsl'
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'adept' --model 'robust'

#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'intphys' --model 'say'
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'intphys' --model 'in' 
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'intphys' --model 'wsl'
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/reduce_dimensionality.py --data 'intphys' --model 'rand'

echo "Done"
