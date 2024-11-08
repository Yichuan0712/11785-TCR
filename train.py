import pickle
from util import printl, printl_file
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F


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
            # distances = []
            similarities = []

            for j, epitope2 in enumerate(epitope_sums.keys()):
                if i == j:
                    continue
                emb2 = epitope_data[epitope2]["average_embedding"].clone().detach()
                # distance = torch.dist(emb1, emb2).item()
                # distances.append((epitope2, distance))
                similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                similarities.append((epitope2, similarity))

            # distances.sort(key=lambda x: x[1])
            # nearest_neighbors[epitope1] = [{"epitope": epitope, "distance": dist} for epitope, dist in
            #                                distances[:N]]
            similarities.sort(key=lambda x: x[1], reverse=True)
            nearest_neighbors[epitope1] = [{"epitope": epitope, "similarity": sim} for epitope, sim in similarities[:N]]

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
            # distances = []
            similarities = []

            for j, epitope2 in enumerate(epitope_sums.keys()):
                if i == j:
                    continue
                emb2 = epitope_data[epitope2]["average_embedding"].clone().detach()
                # distance = torch.dist(emb1, emb2).item()
                # distances.append((epitope2, distance))
                similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                similarities.append((epitope2, similarity))

            # distances.sort(key=lambda x: x[1])
            # nearest_neighbors[epitope1] = [{"epitope": epitope, "distance": dist} for epitope, dist in
            #                                distances[:N]]
            similarities.sort(key=lambda x: x[1], reverse=True)
            nearest_neighbors[epitope1] = [{"epitope": epitope, "similarity": sim} for epitope, sim in similarities[:N]]
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

