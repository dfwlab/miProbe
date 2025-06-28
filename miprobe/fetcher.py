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

import os
from typing import List, Optional
import requests
import numpy as np
import logging
from requests.exceptions import RequestException, Timeout

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://www.biosino.org/iMAC/miProbe/api"
REQUEST_TIMEOUT = 10

# 设置默认请求头，避免 403 错误
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------------------------------------------------------
# Global session for connection pooling
# -----------------------------------------------------------------------------
_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    """Get or create a requests session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        # 设置默认请求头以避免 403 错误
        _session.headers.update(DEFAULT_HEADERS)
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
        if "protein_sequence" not in response_data:
            raise ValueError(f"Invalid response format: missing 'sequence' field")
        
        return response_data["protein_sequence"]
        
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
    headers = {"accept": "application/json"}
    
    if embedding_model:
        params["model_type"] = embedding_model

    try:
        logging.debug("Requesting embedding: %s params=%s", url, params)
        r = session.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
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
