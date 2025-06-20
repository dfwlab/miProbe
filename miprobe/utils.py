"""miProbe Toolkit
=================
A lightweight Python module for accessing the **MiProbe** microbial peptide platform.

Main features
-------------
- Retrieve peptide **sequence** or **embedding** by MiProbe identifier (e.g. `ORF050.00000001`).
- Batch‑download embeddings and wrap them into Numpy arrays, PyTorch Dataset/DataLoader, or tensors for Transformer models.
- Designed for seamless integration with scikit‑learn, PyTorch, TensorFlow, and other ML frameworks.
- Includes an on‑off switch (`USE_REMOTE_API`) so developers can mock responses during offline testing. When set to `True` the module issues real HTTPS requests with a timeout safeguard.

Author  : Dingfeng Wu
Created : 2025‑06‑20
License : MIT
"""
"""
utils.py – Convenience functions for sklearn, torch, transformers
"""

from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from .fetcher import PeptideFetcher
from .dataset import PeptideDataset


def get_numpy_dataset(ids: List[str], embedding_model: str = "prottrans") -> np.ndarray:
    """Return embeddings as a numpy.ndarray suitable for scikit-learn.

    Examples
    --------
    >>> X = get_numpy_dataset(["ORF050.00000001", "ORF050.00000002"])
    >>> X.shape  # (n_samples, dim)
    (2, embedding_dim)
    """
    return PeptideFetcher(embedding_model).batch_fetch_embeddings(ids)


def get_torch_dataloader(
    ids: List[str],
    labels: Optional[List[int]] = None,
    *,
    batch_size: int = 32,
    embedding_model: str = "prottrans",
    shuffle: bool = True,
) -> DataLoader:
    """Return a torch.utils.data.DataLoader for training.

    Parameters
    ----------
    ids : list[str]
        Peptide IDs (MiProbe identifiers).
    labels : list[int] | None
        Supervised classification/regression labels.
    batch_size : int, default 32
        Mini-batch size.
    embedding_model : str, default "prottrans"
        Name of embedding model.
    shuffle : bool, default True
        Whether to shuffle each epoch.
    """
    ds = PeptideDataset(ids, labels, embedding_model)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_transformer_input(ids: List[str], embedding_model: str = "prottrans") -> torch.Tensor:
    """Return embeddings stacked into a single torch.Tensor.

    This is convenient when passing data into Hugging Face Transformer
    heads that accept continuous feature inputs.
    """
    emb = PeptideFetcher(embedding_model).batch_fetch_embeddings(ids)
    return torch.tensor(emb)