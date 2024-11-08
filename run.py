import argparse
import pickle
import yaml
from loss import SupConHardLoss
from util import printl, printl_file
from util import prepare_saving_dir
import torch
import torch.nn as nn
import numpy as np
from box import Box
import sys
from data import get_dataloader, get_dataloader_extraction
from model import prepare_models
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from tqdm import tqdm
import os
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import pandas as pd
from scipy.stats import skew, kurtosis
from train import train_triplet, train_multi
from extract import extract_features
from xgb import xgb_train_and_evaluate


def main(parse_args, configs):
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
    printl(f"Executed with: python {command}", log_path=log_path)
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
    # """
    # Random Seed
    # """
    # if type(configs.fix_seed) == int:
    #     torch.manual_seed(configs.fix_seed)
    #     torch.random.manual_seed(configs.fix_seed)
    #     np.random.seed(configs.fix_seed)
    #     printl(f"{'=' * 128}", log_path=log_path)
    #     printl(f'Random seed set to {configs.fix_seed}.', log_path=log_path)
    """
    Predict?
    """
    if parse_args.mode == 'predict':
        if parse_args.train_feature_path is not None and parse_args.test_feature_path is not None:
            printl(f"{'=' * 128}", log_path=log_path)
            printl(f"XGBoost model training & binding specificity prediction.", log_path=log_path)
            xgb_train_and_evaluate(configs, parse_args.train_feature_path, parse_args.test_feature_path, log_path)
            return
        else:
            raise NotImplementedError
    """
    Dataloader
    """
    printl(f"{'=' * 128}", log_path=log_path)
    dataloaders = get_dataloader(configs, nearest_neighbors=None)
    printl(f'Number of Steps for Training Data: {len(dataloaders["train1_loader"])}', log_path=log_path)
    # printl(f'Number of Steps for Validation Data: {len(dataloaders["valid_loader"])}', log_path=log_path)
    # printl(f'Number of Steps for Test Data: {len(dataloaders_dict["test"])}', log_path=log_path)
    printl("Data loading complete.", log_path=log_path)
    """
    Model
    """
    if parse_args.mode == 'train' and parse_args.resume_path is None:
        printl(f"{'=' * 128}", log_path=log_path)
        encoder, projection_head = prepare_models(configs, log_path=log_path)
        device = torch.device("cuda")
        encoder.to(device)
        projection_head.to(device)
        printl("ESM-2 encoder & projection head initialization complete.", log_path=log_path)
    elif parse_args.mode == 'train' and parse_args.resume_path is not None:
        printl(f"{'=' * 128}", log_path=log_path)
        encoder, projection_head = prepare_models(configs, log_path=log_path)
        device = torch.device("cuda")
        encoder.to(device)
        projection_head.to(device)

        checkpoint = torch.load(parse_args.resume_path, map_location='cuda:0', weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        printl("ESM-2 encoder and projection head successfully resumed from checkpoint.", log_path=log_path)
    elif parse_args.mode == 'extract' and parse_args.resume_path is not None:
        printl(f"{'=' * 128}", log_path=log_path)
        encoder, projection_head = prepare_models(configs, log_path=log_path)
        device = torch.device("cuda")
        encoder.to(device)
        projection_head.to(device)

        checkpoint = torch.load(parse_args.resume_path, map_location='cuda:0', weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        printl("ESM-2 encoder and projection head successfully resumed from checkpoint.", log_path=log_path)
        alphabet = encoder.alphabet
        tokenizer = alphabet.get_batch_converter()
        printl("Tokenizer initialization complete.", log_path=log_path)
        extraction_dataloaders = get_dataloader_extraction(configs)
        printl("Extraction data loading complete.", log_path=log_path)
    else:
        raise NotImplementedError
    """
    Tokenizer, Optimizer, Scheduler, Criterion
    """
    if parse_args.mode == 'train' and parse_args.resume_path is None:
        resume_epoch = 0
        alphabet = encoder.alphabet
        tokenizer = alphabet.get_batch_converter()  # truncation_seq_length=512?
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(projection_head.parameters()),
            lr=float(configs.max_learning_rate),
            betas=(float(configs.optimizer_beta1), float(configs.optimizer_beta2)),
            weight_decay=float(configs.optimizer_weight_decay),
            eps=float(configs.optimizer_eps)
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(configs.scheduler_first_cycle_steps),
            max_lr=float(configs.max_learning_rate),
            min_lr=float(configs.min_learning_rate),
            warmup_steps=int(configs.scheduler_warmup_epochs),
            gamma=float(configs.scheduler_gamma)
        )
        printl("Tokenizer, Optimizer and Scheduler initialization complete.", log_path=log_path)
    elif parse_args.mode == 'train' and parse_args.resume_path is not None:
        resume_epoch = checkpoint['epoch']
        alphabet = encoder.alphabet
        tokenizer = alphabet.get_batch_converter()
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(projection_head.parameters()),
            lr=float(configs.max_learning_rate),
            betas=(float(configs.optimizer_beta1), float(configs.optimizer_beta2)),
            weight_decay=float(configs.optimizer_weight_decay),
            eps=float(configs.optimizer_eps)
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(configs.scheduler_first_cycle_steps),
            max_lr=float(configs.max_learning_rate),
            min_lr=float(configs.min_learning_rate),
            warmup_steps=int(configs.scheduler_warmup_epochs),
            gamma=float(configs.scheduler_gamma)
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        printl("Tokenizer, Optimizer and Scheduler successfully resumed from checkpoint.", log_path=log_path)

    elif parse_args.mode == 'extract' and parse_args.resume_path is not None:
        printl("Start extraction.", log_path=log_path)
        printl(f"{'=' * 128}", log_path=log_path)

        extract_features(encoder, projection_head, extraction_dataloaders["train1_loader"], extraction_dataloaders["train2_loader"], extraction_dataloaders["test_loader"], tokenizer, log_path)

        return
    else:
        raise NotImplementedError

    if configs.contrastive_mode == "Triplet" and parse_args.mode == 'train':
        criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
        printl("Using Triplet Margin Loss.", log_path=log_path)
        printl(f"{'=' * 128}", log_path=log_path)
        nearest_neighbors = None
        for epoch in range(resume_epoch + 1, configs.epochs + 1):
            _nearest_neighbors = train_triplet(encoder, projection_head, epoch, dataloaders["train1_loader"], tokenizer, optimizer, scheduler, criterion, configs, log_path)
            if configs.negative_sampling_mode == 'HardNeg':
                if _nearest_neighbors is not None:
                    nearest_neighbors = _nearest_neighbors
                dataloaders = get_dataloader(configs, nearest_neighbors=nearest_neighbors)
            # save model
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_path, 'model_triplet.pth'))

    elif configs.contrastive_mode == "MultiPosNeg" and parse_args.mode == 'train':
        criterion = SupConHardLoss
        printl("Using SupCon Loss.", log_path=log_path)
        printl(f"{'=' * 128}", log_path=log_path)
        nearest_neighbors = None
        for epoch in range(resume_epoch + 1, configs.epochs + 1):
            _nearest_neighbors = train_multi(encoder, projection_head, epoch, dataloaders["train_loader"], tokenizer, optimizer, scheduler, criterion, configs, log_path)
            if configs.negative_sampling_mode == 'HardNeg':
                if _nearest_neighbors is not None:
                    nearest_neighbors = _nearest_neighbors
                dataloaders = get_dataloader(configs, nearest_neighbors=nearest_neighbors)
            # save model
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_path, 'model_supcon.pth'))
    else:
        raise ValueError("Wrong contrastive mode specified.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ContraTCR: A tool for training and predicting TCR-epitope binding using '
                                                 'contrastive learning.')
    parser.add_argument("--config_path", help="Path to the configuration file. Defaults to "
                                              "'./config/default/config.yaml'. This file contains all necessary "
                                              "parameters and settings for the operation.",
                        default='./config/default/config.yaml')
    parser.add_argument("--mode", help="Operation mode of the script. Use 'train' for training the model, "
                                       "'extract' for feature generation using an existing model, and "
                                       "'predict' for binding specificity prediction. Default mode is "
                                       "'train'.", default='train')
    parser.add_argument("--result_path", default='./result/default/',
                        help="Path where the results will be stored. If not set, results are saved to "
                             "'./result/default/'. This can include prediction outputs or saved models.")
    parser.add_argument("--resume_path", default=None,
                        help="Path to a previously saved model checkpoint. If specified, training or extraction will "
                             "resume from this checkpoint. By default, this is None, meaning training starts from "
                             "scratch.")
    parser.add_argument("--train_feature_path", default=None,
                        help="Path to the input data file. This location has to be specified to be used to "
                             "load data for binding specificity prediction. ")
    parser.add_argument("--test_feature_path", default=None,
                        help="Path to the input data file. This location has to be specified to be used to "
                             "load data for binding specificity prediction. ")

    parse_args = parser.parse_args()

    config_path = parse_args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
        configs = Box(config_dict)

    main(parse_args, configs)
