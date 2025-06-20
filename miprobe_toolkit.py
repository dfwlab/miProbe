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
from __future__ import annotations

from typing import List, Optional
import os
import time
import json
import logging

import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
#: Base REST endpoint for MiProbe
#: Example embedding request:
#:   https://www.biosino.org/iMAC/miProbe/api/embedding?id=ORF050.00000001&format=json
#: Example sequence request  :
#:   https://www.biosino.org/iMAC/miProbe/api/protein?id=ORF050.00000001&format=json
BASE_URL: str = "https://www.biosino.org/iMAC/miProbe/api"

#: Default connection timeout (seconds) for requests.get
REQUEST_TIMEOUT: int | float = 10

#: During local development set to False to avoid HTTP calls. The functions
#: will then return stub data so unit tests can run without external I/O.
USE_REMOTE_API: bool = os.getenv("MIPROBE_USE_REMOTE_API", "0") in {"1", "true", "True"}


# -----------------------------------------------------------------------------
# Helper classes & functions
# -----------------------------------------------------------------------------
class PeptideFetcher:
    """Utility class for downloading peptide sequences and embeddings.

    Parameters
    ----------
    embedding_model : str, default "prottrans"
        Name of the embedding model. Currently MiProbe returns a *default*
        embedding; the parameter is kept for forward compatibility.
    session : requests.Session | None, optional
        Re‑use a session for connection pooling.
    """

    def __init__(self, embedding_model: str = "prottrans", session: requests.Session | None = None):
        self.embedding_model = embedding_model
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_sequence_by_id(self, pid: str) -> str:
        """Return the protein sequence (string of amino‑acid letters).

        During offline testing this returns a stub sequence.
        """
        if not USE_REMOTE_API:
            return "ATCGTGAGA"  # ← stub value for CI / docs builds

        url = f"{BASE_URL}/protein"
        params = {"id": pid, "format": "json"}
        logging.debug("Requesting sequence: %s params=%s", url, params)
        r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()["sequence"]

    def get_embedding_by_id(self, pid: str) -> np.ndarray:
        """Return the peptide embedding as ``np.ndarray`` of dtype ``float32``.
        """
        if not USE_REMOTE_API:
            return np.array([0.1, 0.2, 0.4, 0.1, 0.1, 0.2], dtype=np.float32)

        url = f"{BASE_URL}/embedding"
        params = {"id": pid, "format": "json"}
        if self.embedding_model:
            params["model"] = self.embedding_model

        logging.debug("Requesting embedding: %s params=%s", url, params)
        r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return np.asarray(r.json()["embedding"], dtype=np.float32)

    def batch_fetch_embeddings(self, ids: List[str]) -> np.ndarray:
        """Fetch embeddings for *ids* and stack into ``(n, d)`` array."""
        vectors = [self.get_embedding_by_id(pid) for pid in ids]
        return np.stack(vectors, axis=0)


class PeptideDataset(Dataset):
    """PyTorch ``Dataset`` wrapping peptide embeddings (+ optional labels).

    Parameters
    ----------
    ids : list[str]
        MiProbe identifiers (e.g. ``ORF050.00000001``).
    labels : list[int] | None, optional
        Integer labels for supervised tasks.
    embedding_model : str, default "prottrans"
        Embedding model name forwarded to :class:`PeptideFetcher`.
    """

    def __init__(self, ids: List[str], labels: Optional[List[int]] = None, embedding_model: str = "prottrans"):
        self.fetcher = PeptideFetcher(embedding_model)
        self.ids = ids
        self.labels = labels
        self.embeddings = self.fetcher.batch_fetch_embeddings(ids)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        """Return dataset size."""
        return len(self.ids)

    def __getitem__(self, idx: int):
        """Return one sample.

        Returns ``(x, y)`` if labels are provided, else ``x`` only.
        """
        x = torch.tensor(self.embeddings[idx])
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------

def get_numpy_dataset(ids: List[str], embedding_model: str = "prottrans") -> np.ndarray:
    """Return embeddings as a ``numpy.ndarray`` suitable for scikit‑learn.

    Examples
    --------
    >>> X = get_numpy_dataset(["ORF050.00000001", "ORF050.00000002"])
    >>> X.shape  # (n_samples, dim)
    (2, 6)
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
    """Return a ``torch.utils.data.DataLoader`` for training.

    Parameters
    ----------
    ids : list[str]
        Peptide IDs.
    labels : list[int] | None
        Supervised labels.
    batch_size : int, default 32
        Mini‑batch size.
    embedding_model : str, default "prottrans"
        Name of embedding model.
    shuffle : bool, default True
        Whether to shuffle each epoch.
    """
    ds = PeptideDataset(ids, labels, embedding_model)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_transformer_input(ids: List[str], embedding_model: str = "prottrans") -> torch.Tensor:
    """Return embeddings stacked into a single ``torch.Tensor``.

    This is convenient when passing data into Hugging Face Transformer
    heads that accept continuous inputs.
    """
    emb = PeptideFetcher(embedding_model).batch_fetch_embeddings(ids)
    return torch.tensor(emb)


# -----------------------------------------------------------------------------
# Self‑test / demo (disabled by default for packaging)
# -----------------------------------------------------------------------------
if __name__ == "__main__" and bool(os.getenv("MIPROBE_DEMO", "0")):
    logging.basicConfig(level=logging.INFO)
    example_ids = ["ORF050.00000001", "ORF050.00000002"]
    X = get_numpy_dataset(example_ids)
    print("Embeddings shape:", X.shape)

    dl = get_torch_dataloader(example_ids, batch_size=1, shuffle=False)
    for batch in dl:
        print(batch)
"""
