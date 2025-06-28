import torch
from torch.utils.data import Dataset
from .fetcher import PeptideFetcher

class PeptideDataset(Dataset):
    def __init__(self, ids, labels=None, embedding_model='t5'):
        self.fetcher = PeptideFetcher(embedding_model)
        self.ids = ids
        self.labels = labels
        self.embeddings = self.fetcher.batch_fetch_embeddings(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx])
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x
