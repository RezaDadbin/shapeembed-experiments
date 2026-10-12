import torch
from torch.utils.data import Dataset

from config import DATA_PROCESSED


class DistanceMatrixDataset(Dataset):
    """
    Dataset returning:
        x: normalized distance matrix, shape [1, N, N]
        y: label

    The label is not used to train ShapeEmbed.
    It is saved for downstream classification evaluation.
    """

    def __init__(self, split: str):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.X = torch.load(DATA_PROCESSED / f"X_{split}.pt")
        self.y = torch.load(DATA_PROCESSED / f"y_{split}.pt")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
