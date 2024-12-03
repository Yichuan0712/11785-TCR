#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error_final_multi_regular_lora_33.log
#SBATCH --output=output_final_multi_regular_lora_33.log

#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name RegMuLo33

##SBATCH -p requeue
#SBATCH -p gpu

##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab

module load miniconda3

# Activate the Conda environment
source activate /home/yz3qt/data/miniconda/envs/splm

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python ./run.py --config_path ./config/final/final_multi_regular_lora_33_2.yaml --resume_path ./result/final_multi/20241201-17-25-03/checkpoint/model_supcon.pth --mode train --result_path ./result/final_multi/ || true
