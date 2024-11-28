#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error_final_clean_predict.log
#SBATCH --output=output_final_clean_predict.log

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

python ./run.py --config_path ./config/final/final_clean.yaml --train_feature_path ./result/final_clean_extract/20241128-05-46-31/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-05-46-31/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true

python ./run.py --config_path ./config/final/final_clean_adapter_12.yaml --train_feature_path ./result/final_clean_extract/20241128-05-51-40/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-05-51-40/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true
python ./run.py --config_path ./config/final/final_clean_adapter_16.yaml --train_feature_path ./result/final_clean_extract/20241128-05-56-55/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-05-56-55/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true
python ./run.py --config_path ./config/final/final_clean_adapter_20.yaml --train_feature_path ./result/final_clean_extract/20241128-06-02-10/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-06-02-10/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true

python ./run.py --config_path ./config/final/final_clean_finetune_2.yaml --train_feature_path ./result/final_clean_extract/20241128-06-02-23/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-06-02-23/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true
python ./run.py --config_path ./config/final/final_clean_finetune_4.yaml --train_feature_path ./result/final_clean_extract/20241128-06-07-33/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-06-07-33/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true
python ./run.py --config_path ./config/final/final_clean_finetune_6.yaml --train_feature_path ./result/final_clean_extract/20241128-06-12-43/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-06-12-43/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true

python ./run.py --config_path ./config/final/final_clean_lora_16.yaml --train_feature_path ./result/final_clean_extract/20241128-06-17-53/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-06-17-53/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true
python ./run.py --config_path ./config/final/final_clean_lora_33.yaml --train_feature_path ./result/final_clean_extract/20241128-06-23-06/feature_data_train.csv --test_feature_path ./result/final_clean_extract/20241128-06-23-06/feature_data_test.csv --mode predict --result_path ./result/final_clean_predict/ || true






