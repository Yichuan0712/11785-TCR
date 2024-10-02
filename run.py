import argparse
import yaml
from util import printl, printl_file
from util import prepare_saving_dir
import torch
import numpy as np
from box import Box
import sys
from data import get_dataloader
from model import get_tokenizer, get_model

def main(parse_args, configs, valid_fold_index, test_fold_index):
    torch.cuda.empty_cache()
    curdir_path, result_path, checkpoint_path, log_path, config_path = prepare_saving_dir(parse_args)
    """
    Banner
    """
    printl(f"{'=' * 128}")
    printl("               ______   ______   .__   __. .___________..______          ___   .___________.  ______ .______      ")
    printl("              /      | /  __  \  |  \ |  | |           ||   _  \        /   \  |           | /      ||   _  \     ")
    printl("             |  ,----'|  |  |  | |   \|  | `---|  |----`|  |_)  |      /  ^  \ `---|  |----`|  ,----'|  |_)  |    ")
    printl("             |  |     |  |  |  | |  . `  |     |  |     |      /      /  /_\  \    |  |     |  |     |      /     ")
    printl("             |  `----.|  `--'  | |  |\   |     |  |     |  |\  \----./  _____  \   |  |     |  `----.|  |\  \----.")
    printl("              \______| \______/  |__| \__|     |__|     | _| `._____/__/     \__\  |__|      \______|| _| `._____|")
    printl()
    """
    Description
    """
    printl(f"{'=' * 128}", log_path=log_path)
    printl(configs.description, log_path=log_path)
    """
    CMD
    """
    printl(f"{'=' * 128}", log_path=log_path)
    command = ''.join(sys.argv)
    printl(f"Called with: python {command}", log_path=log_path)
    """
    Directories
    """
    printl(f"{'=' * 128}", log_path=log_path)
    printl(f"Result Directory: {result_path}", log_path=log_path)
    printl(f"Checkpoint Directory: {checkpoint_path}", log_path=log_path)
    printl(f"Log Directory: {log_path}", log_path=log_path)
    printl(f"Config Directory: {config_path}", log_path=log_path)
    printl(f"Current Working Directory: {curdir_path}", log_path=log_path)
    """
    Configration File
    """
    printl(f"{'=' * 128}", log_path=log_path)
    printl_file(parse_args.config_path, log_path=log_path)
    """
    Random Seed
    """
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)
        printl(f"{'=' * 128}", log_path=log_path)
        printl(f'Random seed set to {configs.fix_seed}.', log_path=log_path)
    """
    Fold Split
    """
    printl(f"{'=' * 128}", log_path=log_path)
    all_folds = [0, 1, 2, 3, 4]
    train_folds = [fold for fold in all_folds if fold not in [valid_fold_index, test_fold_index]]
    printl(f"Training Fold Indices: {train_folds}", log_path=log_path)
    printl(f"Validation Fold Index: {valid_fold_index}", log_path=log_path)
    printl(f"Test Fold Index: {test_fold_index}", log_path=log_path)
    """
    Dataloader
    """
    printl(f"{'=' * 128}", log_path=log_path)
    dataloaders_dict = get_dataloader(configs, valid_fold_index, test_fold_index)
    printl(f'Number of Steps for Training Data: {len(dataloaders_dict["train"])}', log_path=log_path)
    printl(f'Number of Steps for Validation Data: {len(dataloaders_dict["valid"])}', log_path=log_path)
    printl(f'Number of Steps for Test Data: {len(dataloaders_dict["test"])}', log_path=log_path)
    printl("Data loading complete.", log_path=log_path)
    """
    Tokenizer
    """
    printl(f"{'=' * 128}", log_path=log_path)
    tokenizer = get_tokenizer(configs)
    printl("Tokenizer initialization complete.", log_path=log_path)
    """
    Model
    """
    printl(f"{'=' * 128}", log_path=log_path)
    encoder = get_model(configs, log_path=log_path)
    printl("Model initialization complete.", log_path=log_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ContraTCR: A tool for training and predicting TCR binding using '
                                                 'contrastive learning.')
    parser.add_argument("--config_path", help="Path to the configuration file. Defaults to "
                                              "'./config/default/config.yaml'. This file contains all necessary "
                                              "parameters and settings for the operation.",
                        default='./config/default/config.yaml')
    parser.add_argument("--mode", help="Operation mode of the script. Use 'train' for training the model and "
                                       "'predict' for making predictions using an existing model. Default mode is "
                                       "'train'.", default='train')
    parser.add_argument("--result_path", default='./result/default/',
                        help="Path where the results will be stored. If not set, results are saved to "
                             "'./result/default/'. This can include prediction outputs or saved models.")
    parser.add_argument("--resume_path", default=None,
                        help="Path to a previously saved model checkpoint. If specified, training or prediction will "
                             "resume from this checkpoint. By default, this is None, meaning training starts from "
                             "scratch.")

    parse_args = parser.parse_args()

    config_path = parse_args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
        configs = Box(config_dict)

    for i in range(1):
        valid_fold_index = i
        if valid_fold_index == 4:
            test_fold_index = 0
        else:
            test_fold_index = valid_fold_index + 1
        main(parse_args, configs, valid_fold_index, test_fold_index)


