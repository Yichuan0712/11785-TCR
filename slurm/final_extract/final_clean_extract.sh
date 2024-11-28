#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error_final_clean_extract.log
#SBATCH --output=output_final_clean_extract.log

#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name ClnX

##SBATCH -p requeue
#SBATCH -p gpu

##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab

module load miniconda3

# Activate the Conda environment
source activate /home/yz3qt/data/miniconda/envs/splm

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python ./run.py --config_path ./config/final/final_clean.yaml --resume_path ./result/final_clean/20241122-18-14-57/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_adapter_12.yaml --resume_path ./result/final_clean/20241124-22-55-47/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_adapter_16.yaml --resume_path ./result/final_clean/20241125-01-48-58/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_adapter_20.yaml --resume_path ./result/final_clean/20241125-02-08-58/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_finetune_2.yaml --resume_path ./result/final_clean/20241125-03-52-40/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_finetune_4.yaml --resume_path ./result/final_clean/20241125-04-50-45/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_finetune_6.yaml --resume_path ./result/final_clean/20241125-07-24-02/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_lora_16.yaml --resume_path ./result/final_clean/20241125-08-36-05/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true
python ./run.py --config_path ./config/final/final_clean_lora_33.yaml --resume_path ./result/final_clean/20241125-10-04-36/checkpoint/model_triplet.pth  --mode extract --result_path ./result/final_clean_extract/ || true






