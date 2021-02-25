#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=discretize
#SBATCH --output=discretize_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/discretize_embedding.py --cache-path '/misc/vlgscratch4/LakeGroup/emin/temporal-prediction/caches_saycam/say_in_256adept.npz' --nbins 64


echo "Done"
