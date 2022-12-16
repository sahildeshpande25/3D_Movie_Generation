#!/bin/bash
#SBATCH --job-name=stereo_pairs_train
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4


singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd $SCRATCH/CV
conda activate cv_proj
python3 train_stereo_pairs.py 
"
