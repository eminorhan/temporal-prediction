#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_dynamics.py --cuda --data 'adept' --embedding-model 'spatial' --batch-size 256 --dropout 0.1
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_dynamics.py --cuda --data 'say' --embedding-model 'say' --batch-size 256 --dropout 0.1
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_dynamics.py --cuda --data 'say' --embedding-model 'wsl' --batch-size 256 --dropout 0.1
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/train_dynamics.py --cuda --data 'say' --embedding-model 'rand' --batch-size 256 --dropout 0.1

echo "Done"
