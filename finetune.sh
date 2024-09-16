#!/bin/bash
#SBATCH --job-name=EAT-HSN_finetuning
#SBATCH --output=/mnt/stud/work/deeplearninglab/ss2024/ssl-1/output/logs/finetuning_test.log
#SBATCH --error=/mnt/stud/work/deeplearninglab/ss2024/ssl-1/output/logs/finetuning_test.err
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1

date;hostname;pwd
source /mnt/stud/work/python/mconda/39/bin/activate base
conda activate /mnt/stud/home/hplutz/miniconda3/envs/birdset

srun python finetune.py
    
echo "Finished script."
date
