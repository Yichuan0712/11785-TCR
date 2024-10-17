import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import random


class PytdcDatasetTriplet(Dataset):
    def __init__(self, dataframe, configs, nearest_neighbors):
        """
        Initializes the PytdcDatasetTriplet dataset object.

        Args:
            dataframe (pd.DataFrame): A DataFrame containing the data to be used in this dataset.
            configs: Configuration parameters that include dataset and model settings.

        This method processes the dataframe to create dictionaries that map TCR sequences to their
        associated epitopes and vice versa, for both positive and negative pairs. It also generates a
        list of unique epitopes for sampling purposes.
        """
        self.configs = configs
        self.nearest_neighbors = nearest_neighbors

        # Using specific columns for features and labels
        if configs.tcr_embedding_source == "BindingSite":
            TCR = dataframe['tcr'].values
        elif configs.tcr_embedding_source == "Full":
            TCR = dataframe['tcr_full'].values
        else:
            raise ValueError("Invalid TCR embedding source specified in configs.")
        epitope = dataframe['epitope_aa'].values
        label = dataframe['label'].values

        # Storing TCR and epitopes based on label, these are positive pairs
        self.TCR = TCR[label == 1]
        self.epitope = epitope[label == 1]

        # These are negative pairs in the original dataset
        self.TCR_neg = TCR[label != 1]
        self.epitope_neg = epitope[label != 1]

        # Generate dictionaries mapping TCR to all related epitopes and vice versa
        self.TCR_epitope = {}
        self.epitope_TCR = {}

        for tcr, epi in zip(self.TCR, self.epitope):
            if tcr not in self.TCR_epitope:
                self.TCR_epitope[tcr] = []
            self.TCR_epitope[tcr].append(epi)

            if epi not in self.epitope_TCR:
                self.epitope_TCR[epi] = []
            self.epitope_TCR[epi].append(tcr)

        # Negative sampling dictionaries
        self.TCR_epitope_neg = {}
        self.epitope_TCR_neg = {}

        for tcr, epi_neg in zip(self.TCR, self.epitope_neg):
            if tcr not in self.TCR_epitope_neg:
                self.TCR_epitope_neg[tcr] = []
            self.TCR_epitope_neg[tcr].append(epi_neg)

        for epi, tcr_neg in zip(self.epitope, self.TCR_neg):
            if epi not in self.epitope_TCR_neg:
                self.epitope_TCR_neg[epi] = []
            self.epitope_TCR_neg[epi].append(tcr_neg)

        self.full_list = []
        if configs.batch_mode == "ByEpitope":
            for ep in self.epitope_TCR.keys():
                self.full_list.append(ep)
        elif configs.batch_mode == "Regular":
            for tcr, epitope in zip(self.TCR, self.epitope):
                self.full_list.append((tcr, epitope))
        else:
            raise ValueError("Invalid batch mode specified in configs.")

    def __len__(self):
        """
        Returns the number of unique epitopes in the dataset.

        Returns:
            int: Length of the number of unique epitopes.
        """
        return len(self.full_list)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample for the triplet-based contrastive learning task.

        Args:
            idx (int): Index for accessing the anchor epitope from the unique epitopes list.

        Returns:
            dict: A dictionary containing 'anchor_TCR', 'positive_TCR', and 'negative_TCR'.
                  These are the sequences involved in the triplet, with the anchor TCR corresponding
                  to the sampled anchor epitope.

        """
        if self.configs.batch_mode == "ByEpitope":
            anchor_epitope = self.full_list[idx]
            anchor_TCR = random.choice(self.epitope_TCR[anchor_epitope])
        elif self.configs.batch_mode == "Regular":
            anchor_TCR, anchor_epitope = self.full_list[idx]
        else:
            raise ValueError("Invalid batch mode specified in configs.")

        positive_TCR = random.choice([tcr for tcr in self.epitope_TCR[anchor_epitope] if tcr != anchor_TCR])

        # Select a negative TCR based on configuration setting
        if self.configs.negative_sampling_mode == 'RandomNeg':
            # Option 1: Randomly select from negative pairs
            negative_TCR_candidates = list(set(self.epitope_TCR_neg.get(anchor_epitope, [])) - {anchor_TCR})
            if not negative_TCR_candidates:
                # Fallback to 'ExcludePos' method if no negative candidates are found
                all_options = set(self.TCR_epitope.keys())
                positive_options = set(self.epitope_TCR[anchor_epitope])
                negative_TCR_candidates = list(all_options - positive_options)
            negative_TCR = random.choice(negative_TCR_candidates)
        elif self.configs.negative_sampling_mode == 'ExcludePos':
            # Option 2: Exclude all positive samples and randomly select
            all_options = set(self.TCR_epitope.keys())
            positive_options = set(self.epitope_TCR[anchor_epitope])
            non_positive_options = list(all_options-positive_options)
            negative_TCR = random.choice(non_positive_options)
        elif self.configs.negative_sampling_mode == 'HardNeg':
            # Option 3: Hard negative samples mining
            all_options = set(self.TCR_epitope.keys())
            positive_options = set(self.epitope_TCR[anchor_epitope])
            neg_options = list(all_options - positive_options)
            if self.nearest_neighbors is not None:
                nearest_list = self.nearest_neighbors.get(anchor_epitope)
                # print(anchor_epitope)
                if nearest_list is not None:
                    neg_epitope_options = [i['epitope'] for i in nearest_list]
                    neg_options = set()
                    for i in neg_epitope_options:
                        neg_options |= set(self.epitope_TCR[i])
                    neg_options.discard(anchor_TCR)
                    neg_options = list(neg_options)
            negative_TCR = random.choice(neg_options)
        else:
            raise ValueError("Invalid negative sampling strategy specified in configs.")
        return {'anchor_epitope': anchor_epitope, 'anchor_TCR': anchor_TCR, 'positive_TCR': positive_TCR, 'negative_TCR': negative_TCR}


class PytdcDatasetMulti(Dataset):
    def __init__(self, dataframe, configs, nearest_neighbors):
        self.configs = configs
        self.nearest_neighbors = nearest_neighbors
        self.n_pos = configs.n_pos
        self.n_neg = configs.n_neg

        # Using specific columns for features and labels
        if configs.tcr_embedding_source == "BindingSite":
            TCR = dataframe['tcr'].values
        elif configs.tcr_embedding_source == "Full":
            TCR = dataframe['tcr_full'].values
        else:
            raise ValueError("Invalid TCR embedding source specified in configs.")
        epitope = dataframe['epitope_aa'].values
        label = dataframe['label'].values

        # Storing TCR and epitopes based on label, these are positive pairs
        self.TCR = TCR[label == 1]
        self.epitope = epitope[label == 1]

        # These are negative pairs in the original dataset
        self.TCR_neg = TCR[label != 1]
        self.epitope_neg = epitope[label != 1]

        # Generate dictionaries mapping TCR to all related epitopes and vice versa
        self.TCR_epitope = {}
        self.epitope_TCR = {}

        for tcr, epi in zip(self.TCR, self.epitope):
            if tcr not in self.TCR_epitope:
                self.TCR_epitope[tcr] = []
            self.TCR_epitope[tcr].append(epi)

            if epi not in self.epitope_TCR:
                self.epitope_TCR[epi] = []
            self.epitope_TCR[epi].append(tcr)

        # Negative sampling dictionaries
        self.TCR_epitope_neg = {}
        self.epitope_TCR_neg = {}

        for tcr, epi_neg in zip(self.TCR, self.epitope_neg):
            if tcr not in self.TCR_epitope_neg:
                self.TCR_epitope_neg[tcr] = []
            self.TCR_epitope_neg[tcr].append(epi_neg)

        for epi, tcr_neg in zip(self.epitope, self.TCR_neg):
            if epi not in self.epitope_TCR_neg:
                self.epitope_TCR_neg[epi] = []
            self.epitope_TCR_neg[epi].append(tcr_neg)

        self.full_list = []
        if configs.batch_mode == "ByEpitope":
            for ep in self.epitope_TCR.keys():
                self.full_list.append(ep)
        elif configs.batch_mode == "Regular":
            for tcr, epitope in zip(self.TCR, self.epitope):
                self.full_list.append((tcr, epitope))
        else:
            raise ValueError("Invalid batch mode specified in configs.")

    def __len__(self):
        """
        Returns the number of unique epitopes in the dataset.

        Returns:
            int: Length of the number of unique epitopes.
        """
        return len(self.full_list)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample for the triplet-based contrastive learning task.

        Args:
            idx (int): Index for accessing the anchor epitope from the unique epitopes list.

        Returns:
            dict: A dictionary containing 'anchor_TCR', 'positive_TCR', and 'negative_TCR'.
                  These are the sequences involved in the triplet, with the anchor TCR corresponding
                  to the sampled anchor epitope.

        """
        if self.configs.batch_mode == "ByEpitope":
            anchor_epitope = self.full_list[idx]
            anchor_TCR = random.choice(self.epitope_TCR[anchor_epitope])
        elif self.configs.batch_mode == "Regular":
            anchor_TCR, anchor_epitope = self.full_list[idx]
        else:
            raise ValueError("Invalid batch mode specified in configs.")

        positive_TCR_candidates = [tcr for tcr in self.epitope_TCR[anchor_epitope] if tcr != anchor_TCR]
        if len(positive_TCR_candidates) < self.n_pos:
            positive_TCR_candidates = positive_TCR_candidates * (self.n_pos // len(positive_TCR_candidates) + 1)
        positive_TCR = random.sample(positive_TCR_candidates, self.n_pos)

        # Select a negative TCR based on configuration setting
        if self.configs.negative_sampling_mode == 'RandomNeg':
            # Option 1: Randomly select from negative pairs
            negative_TCR_candidates = list(set(self.epitope_TCR_neg.get(anchor_epitope, [])) - {anchor_TCR})
            if not negative_TCR_candidates:
                # Fallback to 'ExcludePos' method if no negative candidates are found
                all_options = set(self.TCR_epitope.keys())
                positive_options = set(self.epitope_TCR[anchor_epitope])
                negative_TCR_candidates = list(all_options - positive_options)
            if len(negative_TCR_candidates) < self.n_neg:
                negative_TCR_candidates = negative_TCR_candidates * (self.n_neg // len(negative_TCR_candidates) + 1)
            negative_TCR = random.sample(negative_TCR_candidates, self.n_neg)
        elif self.configs.negative_sampling_mode == 'ExcludePos':
            # Option 2: Exclude all positive samples and randomly select
            all_options = set(self.TCR_epitope.keys())
            positive_options = set(self.epitope_TCR[anchor_epitope])
            non_positive_options = list(all_options - positive_options)
            if len(non_positive_options) < self.n_neg:
                non_positive_options = non_positive_options * (self.n_neg // len(non_positive_options) + 1)
            negative_TCR = random.sample(non_positive_options, self.n_neg)
        elif self.configs.negative_sampling_mode == 'HardNeg':
            # Option 3: Hard negative samples mining
            all_options = set(self.TCR_epitope.keys())
            positive_options = set(self.epitope_TCR[anchor_epitope])
            neg_options = list(all_options - positive_options)
            if self.nearest_neighbors is not None:
                nearest_list = self.nearest_neighbors.get(anchor_epitope)
                # print(anchor_epitope)
                if nearest_list is not None:
                    neg_epitope_options = [i['epitope'] for i in nearest_list]
                    neg_options = set()
                    for i in neg_epitope_options:
                        neg_options |= set(self.epitope_TCR[i])
                    neg_options.discard(anchor_TCR)
                    neg_options = list(neg_options)
            if len(neg_options) < self.n_neg:
                neg_options = neg_options * (self.n_neg // len(neg_options) + 1)
            negative_TCR = random.sample(neg_options, self.n_neg)
        else:
            raise ValueError("Invalid negative sampling strategy specified in configs.")
        # print("anc", anchor_TCR)
        # print("pos", positive_TCR)
        # print("neg", negative_TCR)

        anchor_positive_negative_TCR = [anchor_TCR]
        anchor_positive_negative_TCR.extend(positive_TCR)
        anchor_positive_negative_TCR.extend(negative_TCR)
        # print(anchor_positive_negative_TCR)
        # print(len(anchor_positive_negative_TCR))
        # exit(0)
        return {'anchor_epitope': anchor_epitope, 'anchor_positive_negative_TCR': anchor_positive_negative_TCR}


def preserve_structure_collate_fn(batch):
    return batch


def get_dataloader(configs, nearest_neighbors):
    if configs.dataset == "PyTDC":
        train_data = pd.read_csv(f'./dataset/pytdc/train_PyTDC.csv')
        # valid_data = pd.read_csv(f'./dataset/pytdc/valid_PyTDC.csv')
        if configs.contrastive_mode == "Triplet":
            train_dataset = PytdcDatasetTriplet(train_data, configs, nearest_neighbors)
            # valid_dataset = PytdcDatasetTriplet(valid_data, configs, nearest_neighbors)
        elif configs.contrastive_mode == "MultiPosNeg":
            train_dataset = PytdcDatasetMulti(train_data, configs, nearest_neighbors)
            # valid_dataset = PytdcDatasetMulti(valid_data, configs, nearest_neighbors)
        else:
            raise ValueError("Wrong contrastive mode specified.")
        if configs.batch_mode == "ByEpitope":
            batch_size = len(train_dataset.epitope_TCR.keys())
        elif configs.batch_mode == "Regular":
            batch_size = configs.batch_size
        else:
            raise ValueError("Invalid batch mode specified in configs.")

        if configs.contrastive_mode == "Triplet":
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # , drop_last=True)
            # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        elif configs.contrastive_mode == "MultiPosNeg":
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=preserve_structure_collate_fn)  # , drop_last=True)
            # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=preserve_structure_collate_fn)
        return {'train_loader': train_loader, 'valid_loader': None,  # valid_loader,
                'epitope_TCR': train_dataset.epitope_TCR, 'TCR_epitope': train_dataset.TCR_epitope,
                'epitope_TCR_neg': train_dataset.epitope_TCR_neg, 'TCR_epitope_neg': train_dataset.TCR_epitope_neg}
    else:
        raise ValueError("Wrong dataset specified.")
