import pickle
from util import printl
import torch
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import pandas as pd
from scipy.stats import skew, kurtosis


def infer_features(encoder, projection_head, train1_loader, train2_loader, test_loader, tokenizer, log_path):
    device = torch.device("cuda")

    encoder.eval()
    projection_head.eval()

    progress_bar = tqdm(enumerate(train1_loader), total=len(train1_loader), desc=f"Cluster Center Calculation")

    log_dir = os.path.dirname(log_path)
    log_file_average = os.path.join(log_dir, "epitope_average.pkl")

    epitope_sums = {}
    epitope_counts = {}

    with torch.no_grad():
        for batch, data in progress_bar:
            epitope_list = data['epitope']
            anchor_list = data['TCR']

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

    progress_bar2 = tqdm(enumerate(train2_loader), total=len(train2_loader), desc="Finding Nearest Cluster Centers")
    true_classes = []

    feature_list = []

    with torch.no_grad():
        for batch, data in progress_bar2:
            epitope_list = data['epitope']
            anchor_list = data['TCR']
            epitope_smi_list = data['epitope_smi']
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
                    'smi': epitope_smi_list[i],
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
    csv_path = os.path.join(log_dir, 'feature_data_train.csv')
    feature_df.to_csv(csv_path, index=False)
    printl(f"Training features are saved to {csv_path}.", log_path=log_path)

    progress_bar3 = tqdm(enumerate(test_loader), total=len(test_loader),
                         desc="Finding Nearest Cluster Centers")
    true_classes = []

    feature_list = []

    with torch.no_grad():
        for batch, data in progress_bar3:
            epitope_list = data['epitope']
            anchor_list = data['TCR']
            epitope_smi_list = data['epitope_smi']
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
                similarity_to_own_cluster = F.cosine_similarity(anchor_embs[i].unsqueeze(0),
                                                                target_cluster_emb.unsqueeze(0)).item()

                rank_position = [d[0] for d in cosine_similarities].index(epitope) + 1  # 索引从0开始，故加1

                features = {
                    'x': anchor_list[i],
                    'y': epitope,
                    'smi': epitope_smi_list[i],
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
    csv_path = os.path.join(log_dir, 'feature_data_test.csv')
    feature_df.to_csv(csv_path, index=False)
    printl(f"Test features are saved to {csv_path}.", log_path=log_path)

