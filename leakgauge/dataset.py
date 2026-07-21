import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import random_split
import random
import itertools

class SimpleDataset(Dataset):
    def __init__(self, X, y, device='cuda'):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DatasetManager:
    def __init__(
        self,
        X,
        y,
        batch_size=64,
        test_ratio=0.2,
        target_data_len=50,
        pad_value=-10,
        seed=42,
        device='cuda' 
    ):
        self.max_len = target_data_len
        self.pad_value = pad_value
        self.device = device

        for i in range(len(X)):
            X[i] = self.pad(X[i])

        dataset = SimpleDataset(X, y, device=device)

        total_size = len(dataset)
        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size

        generator = torch.Generator().manual_seed(seed)

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def pad(self, x):
        if len(x) < self.max_len:
            padding = [self.pad_value] * (self.max_len - len(x))
            x = np.concatenate([x, padding])
        return x[:self.max_len]