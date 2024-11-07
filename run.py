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
    printl(f'Number of Steps for Training Data: {len(dataloaders["train_loader"])}', log_path=log_path)
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
    elif parse_args.mode == 'predict' and parse_args.resume_path is not None:
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
    elif parse_args.mode == 'predict' and parse_args.resume_path is not None:
        printl("Start prediction.", log_path=log_path)
        printl(f"{'=' * 128}", log_path=log_path)

        infer_features(encoder, projection_head, dataloaders["train_loader"], tokenizer, inference_dataloaders["valid_loader"], log_path)

        return
    else:
        raise NotImplementedError

    if configs.contrastive_mode == "Triplet" and parse_args.mode == 'train':
        criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
        printl("Using Triplet Margin Loss.", log_path=log_path)
        printl(f"{'=' * 128}", log_path=log_path)
        nearest_neighbors = None
        for epoch in range(resume_epoch + 1, configs.epochs + 1):
            _nearest_neighbors = train_triplet(encoder, projection_head, epoch, dataloaders["train_loader"], tokenizer, optimizer, scheduler, criterion, configs, log_path)
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


def train_triplet(encoder, projection_head, epoch, train_loader, tokenizer, optimizer, scheduler, criterion, configs, log_path):
    device = torch.device("cuda")

    encoder.train()
    projection_head.train()

    total_loss = 0

    if configs.batch_mode == "Regular":
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}]")
    elif configs.batch_mode == "ByEpitope":
        progress_bar = enumerate(train_loader)
    else:
        raise ValueError("Invalid batch mode specified in configs.")

    if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
        log_dir = os.path.dirname(log_path)
        log_file_average = os.path.join(log_dir, "epitope_average.pkl")
        log_file_distance = os.path.join(log_dir, "epitope_distance.pkl")
        epitope_sums = {}
        epitope_counts = {}

    for batch, data in progress_bar:
        epitope_list = data['anchor_epitope']
        anchor_list = data['anchor_TCR']
        positive_list = data['positive_TCR']
        negative_list = data['negative_TCR']

        anchor_seq_batch = [(epitope_list[i], str(anchor_list[i])) for i in range(len(epitope_list))]
        _, _, anchor_tokens = tokenizer(anchor_seq_batch)

        positive_seq_batch = [(epitope_list[i], str(positive_list[i])) for i in range(len(epitope_list))]
        _, _, positive_tokens = tokenizer(positive_seq_batch)

        negative_seq_batch = [(epitope_list[i], str(negative_list[i])) for i in range(len(epitope_list))]
        _, _, negative_tokens = tokenizer(negative_seq_batch)

        anchor_embs = projection_head(encoder(anchor_tokens.to(device)).mean(dim=1))
        positive_embs = projection_head(encoder(positive_tokens.to(device)).mean(dim=1))
        negative_embs = projection_head(encoder(negative_tokens.to(device)).mean(dim=1))

        loss = criterion(anchor_embs, positive_embs, negative_embs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        if configs.batch_mode == "Regular":
            progress_bar.set_postfix(loss=loss.item())

        if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
            for i, epitope in enumerate(epitope_list):
                if epitope not in epitope_sums:
                    epitope_sums[epitope] = anchor_embs[i]
                    epitope_counts[epitope] = 1
                else:
                    epitope_sums[epitope] += anchor_embs[i]
                    epitope_counts[epitope] += 1

    if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
        epitope_data = {
            epitope: {
                "average_embedding": (epitope_sums[epitope] / epitope_counts[epitope]),
                "count": epitope_counts[epitope]
            }
            for epitope in epitope_sums
        }

        N = int(configs.hard_neg_mining_sample_num)
        nearest_neighbors = {}

        for i, epitope1 in enumerate(epitope_sums.keys()):
            emb1 = epitope_data[epitope1]["average_embedding"].clone().detach()
            distances = []

            for j, epitope2 in enumerate(epitope_sums.keys()):
                if i == j:
                    continue
                emb2 = epitope_data[epitope2]["average_embedding"].clone().detach()
                distance = torch.dist(emb1, emb2).item()
                distances.append((epitope2, distance))

            distances.sort(key=lambda x: x[1])
            nearest_neighbors[epitope1] = [{"epitope": epitope, "distance": dist} for epitope, dist in
                                           distances[:N]]

    avg_loss = total_loss / len(train_loader)
    printl(f"Epoch [{epoch}] completed. Average Loss: {avg_loss:.4f}", log_path=log_path)

    if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
        with open(log_file_average, "wb") as f:
            pickle.dump(epitope_data, f)
            # print(len(epitope_data))

        with open(log_file_distance, "wb") as f:
            pickle.dump(nearest_neighbors, f)

        printl(f"Distance map updated.", log_path=log_path)
        return nearest_neighbors

    return None


def train_multi(encoder, projection_head, epoch, train_loader, tokenizer, optimizer, scheduler, criterion, configs, log_path):
    device = torch.device("cuda")

    encoder.train()
    projection_head.train()

    total_loss = 0

    if configs.batch_mode == "Regular":
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}]")
    elif configs.batch_mode == "ByEpitope":
        progress_bar = enumerate(train_loader)
    else:
        raise ValueError("Invalid batch mode specified in configs.")

    if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
        log_dir = os.path.dirname(log_path)
        log_file_average = os.path.join(log_dir, "epitope_average.pkl")
        log_file_distance = os.path.join(log_dir, "epitope_distance.pkl")
        epitope_sums = {}
        epitope_counts = {}

    for batch, data in progress_bar:
        epitope_list = []
        anchor_positive_negative_list = []
        for element in data:
            epitope_list.append(element['anchor_epitope'])
            anc_pos_neg_mini_batch = [(None, str(element['anchor_positive_negative_TCR'][i])) for i in range(len(element['anchor_positive_negative_TCR']))]
            # print(anc_pos_neg_mini_batch)
            _, _, anc_pos_neg_tokens_mini_batch = tokenizer(anc_pos_neg_mini_batch)
            # print(anc_pos_neg_tokens_mini_batch)
            anc_pos_neg_emb_mini_batch = projection_head(encoder(anc_pos_neg_tokens_mini_batch.to(device)).mean(dim=1))
            # print(anc_pos_neg_emb_mini_batch)
            # print(anc_pos_neg_emb_mini_batch.shape)
            # exit(0)
            anchor_positive_negative_list.append(anc_pos_neg_emb_mini_batch)
        anchor_positive_negative = torch.stack(anchor_positive_negative_list)
        # print(len(epitope_list), epitope_list)
        # print(anchor_positive_negative.shape)
        # exit(0)

        loss = criterion(anchor_positive_negative, configs.temp, configs.n_pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        if configs.batch_mode == "Regular":
            progress_bar.set_postfix(loss=loss.item())

        if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
            for i, epitope in enumerate(epitope_list):
                if epitope not in epitope_sums:
                    epitope_sums[epitope] = anchor_positive_negative[i][0]
                    epitope_counts[epitope] = 1
                else:
                    epitope_sums[epitope] += anchor_positive_negative[i][0]
                    epitope_counts[epitope] += 1

    if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
        epitope_data = {
            epitope: {
                "average_embedding": (epitope_sums[epitope] / epitope_counts[epitope]),
                "count": epitope_counts[epitope]
            }
            for epitope in epitope_sums
        }

        N = int(configs.hard_neg_mining_sample_num)
        nearest_neighbors = {}

        for i, epitope1 in enumerate(epitope_sums.keys()):
            emb1 = epitope_data[epitope1]["average_embedding"].clone().detach()
            distances = []

            for j, epitope2 in enumerate(epitope_sums.keys()):
                if i == j:
                    continue
                emb2 = epitope_data[epitope2]["average_embedding"].clone().detach()
                distance = torch.dist(emb1, emb2).item()
                distances.append((epitope2, distance))

            distances.sort(key=lambda x: x[1])
            nearest_neighbors[epitope1] = [{"epitope": epitope, "distance": dist} for epitope, dist in
                                           distances[:N]]

    avg_loss = total_loss / len(train_loader)
    printl(f"Epoch [{epoch}] completed. Average Loss: {avg_loss:.4f}", log_path=log_path)

    if configs.negative_sampling_mode == 'HardNeg' and epoch % configs.hard_neg_mining_adaptive_rate == 0:
        with open(log_file_average, "wb") as f:
            pickle.dump(epitope_data, f)
            # print(len(epitope_data))

        with open(log_file_distance, "wb") as f:
            pickle.dump(nearest_neighbors, f)

        printl(f"Distance map updated.", log_path=log_path)  # 慢, 得查查为什么
        return nearest_neighbors

    return None


def infer_one(encoder, projection_head, train_loader, tokenizer, valid_or_test_loader, log_path):
    device = torch.device("cuda")
    # Set models to evaluation mode for inference
    encoder.eval()
    projection_head.eval()

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Cluster Center Calculation")

    log_dir = os.path.dirname(log_path)
    log_file_average = os.path.join(log_dir, "epitope_average.pkl")

    epitope_sums = {}
    epitope_counts = {}

    with torch.no_grad():
        for batch, data in progress_bar:
            epitope_list = data['anchor_epitope']
            anchor_list = data['anchor_TCR']

            anchor_seq_batch = [(epitope_list[i], str(anchor_list[i])) for i in range(len(epitope_list))]
            _, _, anchor_tokens = tokenizer(anchor_seq_batch)

            anchor_embs = projection_head(encoder(anchor_tokens.to(device)).mean(dim=1))

            for i, epitope in enumerate(epitope_list):
                if epitope not in epitope_sums:
                    epitope_sums[epitope] = anchor_embs[i]
                    epitope_counts[epitope] = 1
                else:
                    epitope_sums[epitope] += anchor_embs[i]
                    epitope_counts[epitope] += 1
    epitope_data = {
        epitope: {
            "average_embedding": (epitope_sums[epitope] / epitope_counts[epitope]),
            "count": epitope_counts[epitope]
        }
        for epitope in epitope_sums
    }

    with open(log_file_average, "wb") as f:
        pickle.dump(epitope_data, f)
        printl(f"Cluster center calculation completed and saved to {log_file_average}.", log_path=log_path)

    progress_bar2 = tqdm(enumerate(valid_or_test_loader), total=len(valid_or_test_loader), desc="Finding Nearest Cluster Centers")
    true_classes = []
    predicted_classes = []

    prediction_probabilities = []

    with torch.no_grad():
        for batch, data in progress_bar2:
            epitope_list = data['epitope']
            anchor_list = data['TCR']

            anchor_seq_batch = [(epitope_list[i], str(anchor_list[i])) for i in range(len(epitope_list))]
            _, _, anchor_tokens = tokenizer(anchor_seq_batch)

            anchor_embs = projection_head(encoder(anchor_tokens.to(device)).mean(dim=1))

            for i, epitope in enumerate(epitope_list):
                true_classes.append(epitope)

                # Calculate distances to all cluster centers and convert to probabilities
                distances = []
                for cluster_epitope, cluster_data in epitope_data.items():
                    cluster_emb = cluster_data["average_embedding"].to(device)
                    distance = torch.dist(anchor_embs[i], cluster_emb).item()
                    distances.append((cluster_epitope, distance))

                # Convert distances to similarity scores (e.g., inverse distance or cosine similarity)
                inverse_distances = torch.tensor([1 / (d[1] + 1e-8) for d in distances])  # Avoid division by zero
                probabilities = F.softmax(inverse_distances, dim=0).cpu().numpy()

                # Get the epitope with the highest probability as prediction

                nearest_epitope = distances[np.argmax(probabilities)][0]
                predicted_classes.append(nearest_epitope)
                prediction_probabilities.append(dict(zip([d[0] for d in distances], probabilities)))

    return true_classes, predicted_classes, prediction_probabilities

    # return true_classes, predicted_classes


def infer_features(encoder, projection_head, train_loader, tokenizer, valid_or_test_loader, log_path):
    device = torch.device("cuda")

    encoder.eval()
    projection_head.eval()

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Cluster Center Calculation")

    log_dir = os.path.dirname(log_path)
    log_file_average = os.path.join(log_dir, "epitope_average.pkl")

    epitope_sums = {}
    epitope_counts = {}

    with torch.no_grad():
        for batch, data in progress_bar:
            epitope_list = data['anchor_epitope']
            anchor_list = data['anchor_TCR']

            anchor_seq_batch = [(epitope_list[i], str(anchor_list[i])) for i in range(len(epitope_list))]
            _, _, anchor_tokens = tokenizer(anchor_seq_batch)

            anchor_embs = projection_head(encoder(anchor_tokens.to(device)).mean(dim=1))

            for i, epitope in enumerate(epitope_list):
                if epitope not in epitope_sums:
                    epitope_sums[epitope] = anchor_embs[i]
                    epitope_counts[epitope] = 1
                else:
                    epitope_sums[epitope] += anchor_embs[i]
                    epitope_counts[epitope] += 1
    epitope_data = {
        epitope: {
            "average_embedding": (epitope_sums[epitope] / epitope_counts[epitope]),
            "count": epitope_counts[epitope]
        }
        for epitope in epitope_sums
    }

    with open(log_file_average, "wb") as f:
        pickle.dump(epitope_data, f)
        printl(f"Cluster center calculation completed and saved to {log_file_average}.", log_path=log_path)

    progress_bar2 = tqdm(enumerate(valid_or_test_loader), total=len(valid_or_test_loader), desc="Finding Nearest Cluster Centers")
    true_classes = []

    feature_list = []

    with torch.no_grad():
        for batch, data in progress_bar2:
            epitope_list = data['epitope']
            anchor_list = data['TCR']
            label_list = data['label']

            anchor_seq_batch = [(epitope_list[i], str(anchor_list[i])) for i in range(len(epitope_list))]
            _, _, anchor_tokens = tokenizer(anchor_seq_batch)

            anchor_embs = projection_head(encoder(anchor_tokens.to(device)).mean(dim=1))

            for i, epitope in enumerate(epitope_list):
                true_classes.append(epitope)

                cosine_similarities = []
                for cluster_epitope, cluster_data in epitope_data.items():
                    cluster_emb = cluster_data["average_embedding"].to(device)
                    cos_sim = F.cosine_similarity(anchor_embs[i].unsqueeze(0), cluster_emb.unsqueeze(0)).item()
                    cosine_similarities.append((cluster_epitope, cos_sim))

                cosine_similarities.sort(key=lambda x: x[1])

                similarity_values = [d[1] for d in cosine_similarities]

                min_similarity = min(similarity_values)
                max_similarity = max(similarity_values)
                avg_similarity = sum(similarity_values) / len(similarity_values)
                median_similarity = np.median(similarity_values)
                std_similarity = np.std(similarity_values)
                skewness_similarity = skew(similarity_values)
                kurtosis_similarity = kurtosis(similarity_values, fisher=True)

                target_cluster_emb = epitope_data[epitope]["average_embedding"].to(device)
                similarity_to_own_cluster = F.cosine_similarity(anchor_embs[i].unsqueeze(0), target_cluster_emb.unsqueeze(0)).item()

                rank_position = [d[0] for d in cosine_similarities].index(epitope) + 1  # 索引从0开始，故加1

                features = {
                    'x': anchor_list[i],
                    'y': epitope,
                    'similarity_to_own_cluster': similarity_to_own_cluster,
                    'max_similarity': max_similarity,
                    'min_similarity': min_similarity,
                    'avg_similarity': avg_similarity,
                    'median_similarity': median_similarity,
                    'std_similarity': std_similarity,
                    'skewness_similarity': skewness_similarity,
                    'kurtosis_similarity': kurtosis_similarity,
                    'rank_position': rank_position,
                    'label': int(label_list[i]),
                }

                feature_list.append(features)

    feature_df = pd.DataFrame(feature_list)
    csv_path = os.path.join(log_dir, 'feature_data.csv')
    feature_df.to_csv(csv_path, index=False)
    printl(f"Features are saved to {csv_path}.", log_path=log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ContraTCR: A tool for training and predicting TCR-epitope binding using '
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

    main(parse_args, configs)
