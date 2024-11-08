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
from data import get_dataloader, get_dataloader_infer
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
from infer import infer_features


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
    elif parse_args.mode == 'infer' and parse_args.resume_path is not None:
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
        inference_dataloaders = get_dataloader_infer(configs)
        printl("Inference data loading complete.", log_path=log_path)
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

    # elif parse_args.mode == 'predict' and parse_args.resume_path is not None:
    #     printl("Start prediction.", log_path=log_path)
    #     printl(f"{'=' * 128}", log_path=log_path)
    #     true_classes, predicted_classes, predicted_scores = infer_one(encoder, projection_head, dataloaders["train_loader"], tokenizer, inference_dataloaders["test_loader"], log_path)
    #     print(true_classes)
    #     print(predicted_classes)
    #
    #     correct_predictions = sum(1 for true, pred in zip(true_classes, predicted_classes) if true == pred)
    #     accuracy = correct_predictions / len(true_classes) if len(true_classes) > 0 else 0
    #
    #     # 获取所有唯一类别
    #     unique_classes = set(true_classes + predicted_classes)
    #
    #     # 初始化 Precision、Recall 和 F1 Score 列表
    #     precision_scores = []
    #     recall_scores = []
    #     f1_scores = []
    #
    #     for label in unique_classes:
    #         # 计算每个类别的 True Positives、False Positives 和 False Negatives
    #         true_positive = sum(
    #             1 for true, pred in zip(true_classes, predicted_classes) if true == label and pred == label)
    #         predicted_positive = sum(1 for pred in predicted_classes if pred == label)
    #         actual_positive = sum(1 for true in true_classes if true == label)
    #
    #         # 计算 Precision
    #         precision = true_positive / predicted_positive if predicted_positive > 0 else 0
    #         precision_scores.append(precision)
    #
    #         # 计算 Recall
    #         recall = true_positive / actual_positive if actual_positive > 0 else 0
    #         recall_scores.append(recall)
    #
    #         # 计算 F1 Score
    #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #         f1_scores.append(f1)
    #
    #     precision = sum(precision_scores) / len(precision_scores)
    #     recall = sum(recall_scores) / len(recall_scores)
    #     f1 = sum(f1_scores) / len(f1_scores)
    #
    #     lb = LabelBinarizer()
    #     true_binarized = lb.fit_transform(true_classes)
    #
    #     # 提取所有类别
    #     all_classes = sorted(set(true_classes + predicted_classes))
    #
    #     # 将 predicted_scores 转换为二维数组，顺序匹配 all_classes
    #     predicted_scores_array = np.array([
    #         [score_dict.get(cls, 0.0) for cls in all_classes]
    #         for score_dict in predicted_scores
    #     ])
    #
    #     # 检查 true_binarized 和 predicted_scores_array 的每一列，移除那些没有正负样本的类别
    #     valid_indices = []
    #     for i in range(true_binarized.shape[1]):
    #         if len(np.unique(true_binarized[:, i])) > 1:  # 确保每个类别至少有一个正样本和一个负样本
    #             valid_indices.append(i)
    #
    #     # 对 true_binarized 和 predicted_scores_array 进行筛选
    #     true_binarized_filtered = true_binarized[:, valid_indices]
    #     predicted_scores_array_filtered = predicted_scores_array[:, valid_indices]
    #
    #     # 计算 AUC
    #     auc = roc_auc_score(true_binarized_filtered, predicted_scores_array_filtered, average="macro",
    #                         multi_class="ovr")
    #
    #     # 显示各项指标
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #     print(f"F1 Score: {f1:.4f}")
    #     print(f"AUC: {auc:.4f}" if auc is not None else "AUC: Not applicable")
    #     return
    elif parse_args.mode == 'infer' and parse_args.resume_path is not None:
        printl("Start inference.", log_path=log_path)
        printl(f"{'=' * 128}", log_path=log_path)

        infer_features(encoder, projection_head, dataloaders["train1_loader"], tokenizer, inference_dataloaders["train2_loader"], log_path)

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
    parser.add_argument("--mode", help="Operation mode of the script. Use 'train' for training the model and "
                                       "'infer' for feature generation using an existing model. Default mode is "
                                       "'train'.", default='train')
    parser.add_argument("--result_path", default='./result/default/',
                        help="Path where the results will be stored. If not set, results are saved to "
                             "'./result/default/'. This can include prediction outputs or saved models.")
    parser.add_argument("--resume_path", default=None,
                        help="Path to a previously saved model checkpoint. If specified, training or inference will "
                             "resume from this checkpoint. By default, this is None, meaning training starts from "
                             "scratch.")

    parse_args = parser.parse_args()

    config_path = parse_args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
        configs = Box(config_dict)

    main(parse_args, configs)
