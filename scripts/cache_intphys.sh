#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=cachint
#SBATCH --output=cachint_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O3'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'say' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O3'

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O3'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'in' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O3'

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O3'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'wsl' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O3'

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'train'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'dev_O3'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O1'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O2'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/cache_intphys.py --model 'rand' --data-dir '/misc/vlgscratch4/LakeGroup/emin/baby-vision-video/intphys_frames/fps_15' --data 'test_O3'

echo "Done"
