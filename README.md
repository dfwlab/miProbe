# miProbe Toolkit

An AI-friendly peptide sequence + embedding database client.

https://miprobe-demo.streamlit.app/

## Key dependencies

Python library versions below match the resolved lockfile (`uv.lock`). The **uv** badge shows a typical CLI version; run `uv --version` on your machine to confirm.

[![uv](https://img.shields.io/badge/uv-0.11.3-A162F7?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/python-≥3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

Install (example): `uv sync`

## Search clients (`miprobe.fetcher`)

All calls use `BASE_URL` (e.g. `https://www.biosino.org/iMAC/miProbe/api` or your test host). Use `MiProbeSearchClient` or the module functions; `run_search_strategies` can chain several steps.

| Kind                          | Idea                                                                                                                             |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding similarity**      | POST body: amino-acid sequence → ProtT5 embedding → FAISS; returns ranked `internal_id` + cosine similarity.                     |
| **Peptide properties**        | Filter `peptide_prop` by length buckets and optional net charge, hydrophobicity, pI, localization, membrane, etc.                |
| **Peptide ↔ bio source link** | Distinct peptide–source pairs from `peptide_source_link` + `bio_source`, with length bins and optional taxonomy/habitat filters. |
| **Bio source search**         | Paginated search on `bio_source` (keyword + field, or faceted filters).                                                          |
| **Taxonomy drill-down**       | Distinct taxonomy values with counts at a chosen level, constrained by parent taxa.                                              |
| **Protein by internal id**    | All `protein_main` rows for one numeric id (optional sequence in the payload).                                                   |
| **Protein properties**        | Full physicochemical record from `peptide_prop` for one id (optionally disambiguate with `seq_length`).                          |
| **Protein sources**           | Genomic / source annotations for one protein via `peptide_source_link` + `bio_source`.                                           |
| **Protein by 50% AAI family** | Paginated `protein_main` listing for one `family_50aai`, with optional length bins and sequence toggle.                          |

Precomputed **embeddings by accession** use `GET …/embedding` with `accession_id` and `model_type` (`prott5` / `esm2`); see `get_embedding_by_id` in the same module.
