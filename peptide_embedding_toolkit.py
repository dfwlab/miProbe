from typing import List, Union, Optional
import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Base URL of the peptide database (assume a RESTful API is available)
BASE_URL = "https://your-peptide-db.org/api"

class PeptideFetcher:
    def __init__(self, embedding_type: str = "prottrans"):
        self.embedding_type = embedding_type

    def get_sequence_by_id(self, pid: str) -> str:
        #response = requests.get(f"{BASE_URL}/sequence/{pid}")
        #response.raise_for_status()
        #return response.json()["sequence"]
        return 'ATCGTGAGA'

    def get_embedding_by_id(self, pid: str) -> np.ndarray:
        #response = requests.get(f"{BASE_URL}/embedding/{self.embedding_type}/{pid}")
        #response.raise_for_status()
        #return np.array(response.json()["embedding"], dtype=np.float32)
        return np.array([0.1, 0.2, 0.4, 0.1, 0.1, 0.2])

    def batch_fetch_embeddings(self, ids: List[str]) -> np.ndarray:
        embeddings = [self.get_embedding_by_id(pid) for pid in ids]
        return np.stack(embeddings)

class PeptideDataset(Dataset):
    def __init__(self, ids: List[str], labels: Optional[List[int]] = None, embedding_type: str = "prottrans"):
        self.fetcher = PeptideFetcher(embedding_type)
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
        else:
            return x

# Utility function to convert to scikit-learn format
def get_numpy_dataset(ids: List[str], embedding_type: str = "prottrans") -> np.ndarray:
    fetcher = PeptideFetcher(embedding_type)
    return fetcher.batch_fetch_embeddings(ids)

# Utility function to get PyTorch DataLoader
def get_torch_dataloader(ids: List[str], labels: Optional[List[int]] = None, batch_size: int = 32,
                         embedding_type: str = "prottrans", shuffle: bool = True) -> DataLoader:
    dataset = PeptideDataset(ids, labels, embedding_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Transformer-ready: placeholder for future integration
def get_transformer_input(ids: List[str], embedding_type: str = "prottrans") -> torch.Tensor:
    fetcher = PeptideFetcher(embedding_type)
    embeddings = fetcher.batch_fetch_embeddings(ids)
    return torch.tensor(embeddings)

# Example usage (commented out for packaging)
# if __name__ == '__main__':
#     ids = ["PEP0001", "PEP0002"]
#     dataloader = get_torch_dataloader(ids)
#     for x in dataloader:
#         print(x)
