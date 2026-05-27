import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(self, source_data, target_data):

        self.source_data = source_data
        self.target_data = target_data

    def __len__(self):

        return len(self.source_data)

    def __getitem__(self, idx):

        source = self.source_data[idx]
        target = self.target_data[idx]

        return source, target