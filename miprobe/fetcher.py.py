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
fetcher.py – Fetch peptide sequences and embeddings from the miProbe platform.
"""

import os
import requests
import numpy as np
import logging

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_URL = "https://www.biosino.org/iMAC/miProbe/api"
REQUEST_TIMEOUT = 10
USE_REMOTE_API = os.getenv("MIPROBE_USE_REMOTE_API", "0") in {"1", "true", "True"}

# -----------------------------------------------------------------------------
# PeptideFetcher Class
# -----------------------------------------------------------------------------
class PeptideFetcher:
    """
    Utility class for downloading peptide sequences and embeddings.

    Parameters
    ----------
    embedding_model : str, default "prottrans"
        Name of the embedding model. Currently MiProbe returns a *default*
        embedding; the parameter is kept for forward compatibility.
    session : requests.Session | None, optional
        Re-use a session for connection pooling.
    """

    def __init__(self, embedding_model: str = "prottrans", session: requests.Session | None = None):
        self.embedding_model = embedding_model
        self.session = session or requests.Session()

    def get_sequence_by_id(self, pid: str) -> str:
        """Return the protein sequence (string of amino-acid letters).

        During offline testing this returns a stub sequence.
        """
        if not USE_REMOTE_API:
            return "ATCGTGAGA"  # stub value for CI / docs builds

        url = f"{BASE_URL}/protein"
        params = {"id": pid, "format": "json"}
        logging.debug("Requesting sequence: %s params=%s", url, params)
        r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()["sequence"]

    def get_embedding_by_id(self, pid: str) -> np.ndarray:
        """Return the peptide embedding as a numpy.ndarray of dtype float32."""
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

    def batch_fetch_embeddings(self, ids):
        """Fetch embeddings for multiple peptide IDs and return stacked array."""
        return np.stack([self.get_embedding_by_id(pid) for pid in ids], axis=0)