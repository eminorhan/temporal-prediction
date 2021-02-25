#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=trainemb
#SBATCH --output=trainemb_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_embedding.py --dataset 'adept' --resume 'in' --n_out 1000 '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_embedding.py --dataset 'adept' --resume 'say' --n_out 1000 '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_embedding.py --dataset 'adept' --resume 'rand' --n_out 1000 '/misc/vlgscratch4/LakeGroup/emin/ADEPT/train'

echo "Done"
