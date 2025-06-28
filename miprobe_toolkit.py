"""miProbe Toolkit
=================
A lightweight Python module for accessing the **MiProbe** microbial peptide platform.

Main features
-------------
- Retrieve peptide **sequence** or **embedding** by MiProbe identifier (e.g. `ORF050.00000001`).
- Batch‑download embeddings and wrap them into Numpy arrays, PyTorch Dataset/DataLoader, or tensors for Transformer models.
- Designed for seamless integration with scikit‑learn, PyTorch, TensorFlow, and other ML frameworks.

Author  : Dingfeng Wu
Created : 2025‑06‑20
License : MIT
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import os
import logging

import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from requests.exceptions import RequestException, Timeout

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://www.biosino.org/iMAC/miProbe/api"
REQUEST_TIMEOUT = 10

# -----------------------------------------------------------------------------
# Global session for connection pooling
# -----------------------------------------------------------------------------
_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    """Get or create a requests session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def get_sequence_by_id(pid: str, session: Optional[requests.Session] = None) -> str:
    """Return the protein sequence (string of amino-acid letters).

    Parameters
    ----------
    pid : str
        MiProbe identifier (e.g. 'ORF050.00000001')
    session : requests.Session | None, optional
        Re-use a session for connection pooling

    Returns
    -------
    str
        Protein sequence as amino-acid letters

    Raises
    ------
    requests.RequestException
        If the API request fails
    ValueError
        If the response format is invalid
    """
    if not pid:
        raise ValueError("Peptide ID cannot be empty")

    session = session or _get_session()
    url = f"{BASE_URL}/protein"
    params = {"id": pid, "data_type": "protein_sequence", "format": "json"}
    
    try:
        logging.debug("Requesting sequence: %s params=%s", url, params)
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        
        response_data = r.json()
        if "sequence" not in response_data:
            raise ValueError(f"Invalid response format: missing 'sequence' field")
        
        return response_data["sequence"]
        
    except Timeout:
        logging.error("Timeout while fetching sequence for ID: %s", pid)
        raise
    except RequestException as e:
        logging.error("Request failed for sequence ID %s: %s", pid, str(e))
        raise
    except ValueError as e:
        logging.error("Invalid response format for sequence ID %s: %s", pid, str(e))
        raise

def get_embedding_by_id(
    pid: str, 
    embedding_model: str = "t5", 
    session: Optional[requests.Session] = None
) -> np.ndarray:
    """Return the peptide embedding as a numpy.ndarray of dtype float32.

    Parameters
    ----------
    pid : str
        MiProbe identifier (e.g. 'ORF050.00000001')
    embedding_model : str, default "t5"
        Name of the embedding model. Currently MiProbe returns a *default*
        embedding; the parameter is kept for forward compatibility.
    session : requests.Session | None, optional
        Re-use a session for connection pooling

    Returns
    -------
    np.ndarray
        Peptide embedding as float32 array

    Raises
    ------
    requests.RequestException
        If the API request fails
    ValueError
        If the response format is invalid
    """
    if not pid:
        raise ValueError("Peptide ID cannot be empty")

    session = session or _get_session()
    url = f"{BASE_URL}/embedding"
    params = {"id": pid, "format": "json"}
    
    if embedding_model:
        params["model"] = embedding_model

    try:
        logging.debug("Requesting embedding: %s params=%s", url, params)
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        
        response_data = r.json()
        if "embedding" not in response_data:
            raise ValueError(f"Invalid response format: missing 'embedding' field")
        
        return np.asarray(response_data["embedding"], dtype=np.float32)
        
    except Timeout:
        logging.error("Timeout while fetching embedding for ID: %s", pid)
        raise
    except RequestException as e:
        logging.error("Request failed for embedding ID %s: %s", pid, str(e))
        raise
    except ValueError as e:
        logging.error("Invalid response format for embedding ID %s: %s", pid, str(e))
        raise

def batch_fetch_embeddings(
    ids: List[str], 
    embedding_model: str = "t5", 
    session: Optional[requests.Session] = None
) -> np.ndarray:
    """Fetch embeddings for multiple peptide IDs and return stacked array.

    Parameters
    ----------
    ids : List[str]
        List of MiProbe identifiers
    embedding_model : str, default "t5"
        Name of the embedding model
    session : requests.Session | None, optional
        Re-use a session for connection pooling

    Returns
    -------
    np.ndarray
        Stacked embeddings array with shape (n_ids, embedding_dim)

    Raises
    ------
    ValueError
        If ids list is empty or contains invalid IDs
    """
    if not ids:
        raise ValueError("IDs list cannot be empty")
    
    session = session or _get_session()
    embeddings = []
    
    for pid in ids:
        try:
            embedding = get_embedding_by_id(pid, embedding_model, session)
            embeddings.append(embedding)
        except Exception as e:
            logging.error("Failed to fetch embedding for ID %s: %s", pid, str(e))
            raise
    
    return np.stack(embeddings, axis=0)

# -----------------------------------------------------------------------------
# PyTorch Dataset
# -----------------------------------------------------------------------------

class PeptideDataset(Dataset):
    """PyTorch Dataset wrapping peptide embeddings (+ optional labels).

    Parameters
    ----------
    ids : List[str]
        MiProbe identifiers (e.g. 'ORF050.00000001').
    labels : List[int] | None, optional
        Integer labels for supervised tasks.
    embedding_model : str, default "t5"
        Embedding model name forwarded to the fetcher functions.
    session : requests.Session | None, optional
        Re-use a session for connection pooling.
    """

    def __init__(
        self, 
        ids: List[str], 
        labels: Optional[List[int]] = None, 
        embedding_model: str = "t5",
        session: Optional[requests.Session] = None
    ):
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        if labels is not None and len(ids) != len(labels):
            raise ValueError("IDs and labels must have the same length")
        
        self.ids = ids
        self.labels = labels
        self.embedding_model = embedding_model
        self.session = session
        
        # Fetch all embeddings at initialization
        self.embeddings = batch_fetch_embeddings(ids, embedding_model, session)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return one sample.

        Returns
        -------
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            Returns (x, y) if labels are provided, else x only.
        """
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x

# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def get_numpy_dataset(ids: List[str], embedding_model: str = "t5") -> np.ndarray:
    """Return embeddings as a numpy.ndarray suitable for scikit-learn.

    Parameters
    ----------
    ids : List[str]
        List of MiProbe identifiers
    embedding_model : str, default "t5"
        Name of the embedding model

    Returns
    -------
    np.ndarray
        Embeddings array with shape (n_samples, embedding_dim)

    Examples
    --------
    >>> X = get_numpy_dataset(["ORF050.00000001", "ORF050.00000002"])
    >>> X.shape  # (n_samples, dim)
    (2, 1024)
    """
    return batch_fetch_embeddings(ids, embedding_model)

def get_torch_dataloader(
    ids: List[str],
    labels: Optional[List[int]] = None,
    *,
    batch_size: int = 32,
    embedding_model: str = "t5",
    shuffle: bool = True,
    session: Optional[requests.Session] = None,
) -> DataLoader:
    """Return a torch.utils.data.DataLoader for training.

    Parameters
    ----------
    ids : List[str]
        Peptide IDs.
    labels : List[int] | None, optional
        Supervised labels.
    batch_size : int, default 32
        Mini-batch size.
    embedding_model : str, default "t5"
        Name of embedding model.
    shuffle : bool, default True
        Whether to shuffle each epoch.
    session : requests.Session | None, optional
        Re-use a session for connection pooling.

    Returns
    -------
    DataLoader
        PyTorch DataLoader for training.
    """
    ds = PeptideDataset(ids, labels, embedding_model, session)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_transformer_input(ids: List[str], embedding_model: str = "t5") -> torch.Tensor:
    """Return embeddings stacked into a single torch.Tensor.

    This is convenient when passing data into Hugging Face Transformer
    heads that accept continuous inputs.

    Parameters
    ----------
    ids : List[str]
        List of MiProbe identifiers
    embedding_model : str, default "t5"
        Name of the embedding model

    Returns
    -------
    torch.Tensor
        Embeddings tensor with shape (n_samples, embedding_dim)
    """
    emb = batch_fetch_embeddings(ids, embedding_model)
    return torch.tensor(emb, dtype=torch.float32)

# -----------------------------------------------------------------------------
# Backward compatibility - PeptideFetcher class
# -----------------------------------------------------------------------------

class PeptideFetcher:
    """
    Utility class for downloading peptide sequences and embeddings.
    
    Note: This class is maintained for backward compatibility.
    Consider using the functional API instead.
    """

    def __init__(self, embedding_model: str = "t5", session: Optional[requests.Session] = None):
        self.embedding_model = embedding_model
        self.session = session

    def get_sequence_by_id(self, pid: str) -> str:
        """Return the protein sequence (string of amino-acid letters)."""
        return get_sequence_by_id(pid, self.session)

    def get_embedding_by_id(self, pid: str) -> np.ndarray:
        """Return the peptide embedding as a numpy.ndarray of dtype float32."""
        return get_embedding_by_id(pid, self.embedding_model, self.session)

    def batch_fetch_embeddings(self, ids: List[str]) -> np.ndarray:
        """Fetch embeddings for multiple peptide IDs and return stacked array."""
        return batch_fetch_embeddings(ids, self.embedding_model, self.session)

# -----------------------------------------------------------------------------
# Demo (disabled by default for packaging)
# -----------------------------------------------------------------------------
if __name__ == "__main__" and bool(os.getenv("MIPROBE_DEMO", "0")):
    logging.basicConfig(level=logging.INFO)
    example_ids = ["ORF050.00000001", "ORF050.00000002"]
    
    try:
        X = get_numpy_dataset(example_ids)
        print("Embeddings shape:", X.shape)

        dl = get_torch_dataloader(example_ids, batch_size=1, shuffle=False)
        for batch in dl:
            print(batch)
    except Exception as e:
        print(f"Demo failed: {e}")
