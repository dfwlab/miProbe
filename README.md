# miProbe Toolkit

An AI-friendly peptide sequence + embedding database client.

[https://miprobe-demo.streamlit.app/](https://miprobe-demo.streamlit.app/)

## Key dependencies

Python library versions below match the resolved lockfile (`uv.lock`). The **uv** badge shows a typical CLI version; run `uv --version` on your machine to confirm.

[uv](https://github.com/astral-sh/uv)
[Python](https://www.python.org/)
[PyTorch](https://pytorch.org/)
[scikit-learn](https://scikit-learn.org/)
[Streamlit](https://streamlit.io/)

Install (example): `uv sync`

## Search clients (`miprobe.fetcher`)

All calls use `BASE_URL` (e.g. `https://www.biosino.org/iMAC/miProbe/api` or test host in development). Use `MiProbeSearchClient` or the module functions; `run_search_strategies` can chain several steps.

You can use the [example.ipynb](example.ipynb) Jupyter notebook to test the API.

| Kind                                       | Idea                                                                                                                                    |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Search Peptide By Embedding Similarity** | POST body: amino-acid sequence → ProtT5 embedding → FAISS; returns ranked `internal_id` + cosine similarity.                            |
| **Search Peptide By Properties**           | Search peptides by physicochemical and related features (length buckets, net charge, hydrophobicity, pI, localization, membrane, etc.). |
| **Search Peptide By Bio Source Link**      | Distinct peptide–source pairs, with length bins and optional taxonomy or habitat filters.                                               |
| **Search Bio Source**                      | Paginated search over biological sources (keyword + field, or faceted filters).                                                         |
| **Search Taxonomy**                        | Distinct taxonomy values with counts at a chosen level, constrained by parent taxa.                                                     |
| **Get Peptide By Internal Id**             | All sequence records for one numeric `internal_id` (optional sequence in the payload).                                                  |
| **Get Peptide Properties By Internal Id**  | Fetch physicochemical properties for one `internal_id` (optionally disambiguate with `seq_length`).                                     |
| **Get Peptide Sources By Internal Id**     | Genomic and sample-level source annotations for one peptide, linked from peptide to biological origin.                                  |
| **Get Peptide By 50% AAI Family**          | Paginated listing for one `family_50aai`, with optional length bins and sequence toggle.                                                |

Precomputed **embeddings by accession** use `GET …/embedding` with `accession_id` and `model_type` (`prott5` / `esm2`); see `get_embedding_by_id` in the same module.
