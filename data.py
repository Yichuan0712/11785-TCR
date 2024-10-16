import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import random


class PytdcDatasetTriplet(Dataset):
    def __init__(self, dataframe, configs):
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

        # Using specific columns for features and labels
        if configs.sequence_mode == "BindingSite":
            TCR = dataframe['tcr'].values
        elif configs.sequence_mode == "Full":
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
            negative_TCR = random.choice(self.epitope_TCR_neg[anchor_epitope])
        elif self.configs.negative_sampling_mode == 'ExcludePos':
            # Option 2: Exclude all positive samples and randomly select
            all_options = set(self.TCR_epitope.keys())
            positive_options = set(self.epitope_TCR[anchor_epitope])
            non_positive_options = list(all_options - positive_options)
            negative_TCR = random.choice(non_positive_options)
        else:
            raise ValueError("Invalid negative sampling strategy specified in configs.")
        return {'anchor_epitope': anchor_epitope, 'anchor_TCR': anchor_TCR, 'positive_TCR': positive_TCR, 'negative_TCR': negative_TCR}


class PytdcDatasetMulti(Dataset):
    def __init__(self, dataframe, configs):
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

        # Using specific columns for features and labels
        if configs.sequence_mode == "BindingSite":
            TCR = dataframe['tcr'].values
        elif configs.sequence_mode == "Full":
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
            negative_TCR = random.choice(self.epitope_TCR_neg[anchor_epitope])
        elif self.configs.negative_sampling_mode == 'ExcludePos':
            # Option 2: Exclude all positive samples and randomly select
            all_options = set(self.TCR_epitope.keys())
            positive_options = set(self.epitope_TCR[anchor_epitope])
            non_positive_options = list(all_options - positive_options)
            negative_TCR = random.choice(non_positive_options)
        else:
            raise ValueError("Invalid negative sampling strategy specified in configs.")
        return {'anchor_epitope': anchor_epitope, 'anchor_TCR': anchor_TCR, 'positive_TCR': positive_TCR, 'negative_TCR': negative_TCR}


def get_dataloader(configs):
    if configs.dataset == "PyTDC":
        train_data = pd.read_csv(f'./dataset/pytdc/train_PyTDC.csv')
        valid_data = pd.read_csv(f'./dataset/pytdc/valid_PyTDC.csv')
        if configs.contrastive_mode == "Triplet":
            train_dataset = PytdcDatasetTriplet(train_data, configs)
            valid_dataset = PytdcDatasetTriplet(valid_data, configs)
        else:
            raise ValueError("Wrong contrastive mode specified.")
        if configs.batch_mode == "ByEpitope":
            batch_size = len(train_dataset.epitope_TCR.keys())
        elif configs.batch_mode == "Regular":
            batch_size = configs.batch_size
        else:
            raise ValueError("Invalid batch mode specified in configs.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # , drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return {'train_loader': train_loader, 'valid_loader': valid_loader,
                'epitope_TCR': train_dataset.epitope_TCR, 'TCR_epitope': train_dataset.TCR_epitope,
                'epitope_TCR_neg': train_dataset.epitope_TCR_neg, 'TCR_epitope_neg': train_dataset.TCR_epitope_neg}
    else:
        raise ValueError("Wrong dataset specified.")
