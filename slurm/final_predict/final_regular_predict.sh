#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error_final_regular_predict.log
#SBATCH --output=output_final_regular_predict.log

#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name RegP

##SBATCH -p requeue
#SBATCH -p gpu

##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab

module load miniconda3

# Activate the Conda environment
source activate /home/yz3qt/data/miniconda/envs/splm

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python ./run.py --config_path ./config/final/final_regular.yaml --train_feature_path ./result/final_regular_extract/20241128-04-59-19/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-04-59-19/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true

python ./run.py --config_path ./config/final/final_regular_adapter_12.yaml --train_feature_path ./result/final_regular_extract/20241128-05-04-27/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-04-27/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true
python ./run.py --config_path ./config/final/final_regular_adapter_16.yaml --train_feature_path ./result/final_regular_extract/20241128-05-09-40/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-09-40/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true
python ./run.py --config_path ./config/final/final_regular_adapter_20.yaml --train_feature_path ./result/final_regular_extract/20241128-05-14-55/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-14-55/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true

python ./run.py --config_path ./config/final/final_regular_finetune_2.yaml --train_feature_path ./result/final_regular_extract/20241128-05-20-13/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-20-13/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true
python ./run.py --config_path ./config/final/final_regular_finetune_4.yaml --train_feature_path ./result/final_regular_extract/20241128-05-25-24/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-25-24/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true
python ./run.py --config_path ./config/final/final_regular_finetune_6.yaml --train_feature_path ./result/final_regular_extract/20241128-05-30-35/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-30-35/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true

python ./run.py --config_path ./config/final/final_regular_lora_16.yaml --train_feature_path ./result/final_regular_extract/20241128-05-35-47/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-35-47/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true
python ./run.py --config_path ./config/final/final_regular_lora_33.yaml --train_feature_path ./result/final_regular_extract/20241128-05-40-59/feature_data_train.csv --test_feature_path ./result/final_regular_extract/20241128-05-40-59/feature_data_test.csv --mode predict --result_path ./result/final_regular_predict/ || true






