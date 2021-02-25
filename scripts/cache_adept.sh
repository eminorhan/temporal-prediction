#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=cachadept
#SBATCH --output=cachadept_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_adept.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train' 
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_adept.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train'
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_adept.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_adept.py --model 'spatial' --data-dir '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train'


echo "Done"
