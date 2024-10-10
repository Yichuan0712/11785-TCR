import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd


class PytdcDatasetTriplet(Dataset):
    def __init__(self, dataframe, configs):
        """
        Initializes the dataset object.
        :param dataframe: A DataFrame containing the data to be used in this dataset.
        """
        self.configs = configs

        # Using specific columns for features and labels
        TCR = dataframe['tcr'].values
        epitope = dataframe['epitope_aa'].values
        label = dataframe['label'].values

        # Storing TCR and epitope based on label
        self.TCR = TCR[label == 1]
        self.epitope = epitope[label == 1]

        self.TCR_neg = TCR[label != 1]
        self.epitope_neg = epitope[label != 1]

        # TODO get TCR_epitope, epitope_TCR
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.epitope)

    def __getitem__(self, idx):
        """
        Retrieves the feature tensor and label tensor at the specified index.
        :param idx: Index of the data point to retrieve.
        :return: A tuple containing the feature tensor and label tensor.
        """
        return self.TCR[idx], self.epitope[idx]


def get_dataloader(configs):
    if configs.dataset == "pytdc":
        train_data = pd.read_csv(f'./dataset/pytdc/train_PyTDC.csv')
        valid_data = pd.read_csv(f'./dataset/pytdc/valid_PyTDC.csv')
        if configs.contrastive_mode == "Triplet":
            train_dataset = PytdcDatasetTriplet(train_data, configs)
            valid_dataset = PytdcDatasetTriplet(valid_data, configs)
            # get
            # get
        else:
            raise ValueError("Wrong contrastive mode specified.")
        train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size, shuffle=False)
        return {'train': train_loader, 'valid': valid_loader}  # get
    else:
        raise ValueError("Wrong dataset specified.")
