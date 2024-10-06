#!/bin/bash
#SBATCH --job-name=EAT-NBP
#SBATCH --output=./finetuning_NBP.log
#SBATCH --error=./finetuning_NBP.err
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1

date;hostname;pwd
source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate birdset

srun python finetune.py
    
echo "Finished script."
date
