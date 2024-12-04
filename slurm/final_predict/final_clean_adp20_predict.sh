#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error_final_clean_adp20_predict.log
#SBATCH --output=output_final_clean_adp20_predict.log

#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name ClnP

##SBATCH -p requeue
#SBATCH -p gpu

##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab

module load miniconda3

# Activate the Conda environment
source activate /home/yz3qt/data/miniconda/envs/splm

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python ./run.py --config_path ./config/final/final_clean_adapter_20_2.yaml --train_feature_path ./result/final_clean_extract/20241204-16-53-18/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241204-16-53-18/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true





