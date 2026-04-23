"""miProbe Toolkit
=================
A lightweight Python module for accessing the **MiProbe** microbial peptide platform.

Main features
-------------
- Retrieve peptide **embedding** by accession id (e.g. `ORF050.00000001`) via the precomputed embedding API.
- Batch‑download embeddings and wrap them into Numpy arrays, PyTorch Dataset/DataLoader, or tensors for Transformer models.
- Designed for seamless integration with scikit‑learn, PyTorch, TensorFlow, and other ML frameworks.

Author  : Dingfeng Wu
Created : 2025‑06‑20
Latest Modified by: Peigen Yao @ 2026‑04‑23
License : MIT
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import requests
from requests.exceptions import RequestException, Timeout

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


BASE_URL = "https://www.biosino.org/iMAC/miProbe/api"


# Default for typical GETs and precomputed embedding fetches.
REQUEST_TIMEOUT = 60
# POST /embedding-search/search runs ProtT5 + FAISS; responses often exceed a short read window.
EMBEDDING_SEARCH_TIMEOUT = 120
REQUEST_NUMBER_LIMIT = 10

# Set default headers to avoid 403 errors
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}

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


def _url(path: str) -> str:
    p = path if path.startswith("/") else f"/{path}"
    return f"{BASE_URL}{p}"


def _prune_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    return {k: v for k, v in params.items() if v is not None}


# -----------------------------------------------------------------------------
# Peptide property filters (search-peptide-prop): human-readable labels -> API int codes
# -----------------------------------------------------------------------------

LOCALIZATION_LABEL_TO_CODE: Dict[str, int] = {
    "Others": 0,
    "Cell-Membrane": 1,
    "Cytoplasm": 2,
    "Extracellular": 3,
}
LOCALIZATION_CODE_TO_LABEL: Dict[int, str] = {
    v: k for k, v in LOCALIZATION_LABEL_TO_CODE.items()
}
_VALID_LOCALIZATION_CODES = frozenset(LOCALIZATION_CODE_TO_LABEL)

MEMBRANE_LABEL_TO_CODE: Dict[str, int] = {
    "Membrane bound": 0,
    "Soluble": 1,
}
MEMBRANE_CODE_TO_LABEL: Dict[int, str] = {
    v: k for k, v in MEMBRANE_LABEL_TO_CODE.items()
}
_VALID_MEMBRANE_CODES = frozenset(MEMBRANE_CODE_TO_LABEL)


def _parse_localization_for_api(
    value: Optional[Union[int, str]],
) -> Optional[int]:
    """Map localization label (exact string) or int code to API int. None passes through."""
    if value is None:
        return None
    if isinstance(value, int):
        if value not in _VALID_LOCALIZATION_CODES:
            raise ValueError(
                "localization int must be in "
                f"{sorted(_VALID_LOCALIZATION_CODES)}, got {value!r}"
            )
        return value
    if isinstance(value, str):
        if value not in LOCALIZATION_LABEL_TO_CODE:
            raise ValueError(
                "localization str must be exactly one of "
                f"{list(LOCALIZATION_LABEL_TO_CODE.keys())}, got {value!r}"
            )
        return LOCALIZATION_LABEL_TO_CODE[value]
    raise TypeError(
        f"localization must be int, str, or None, got {type(value).__name__}"
    )


def _parse_membrane_for_api(value: Optional[Union[int, str]]) -> Optional[int]:
    """Map membrane label (exact string) or int code to API int. None passes through."""
    if value is None:
        return None
    if isinstance(value, int):
        if value not in _VALID_MEMBRANE_CODES:
            raise ValueError(
                f"membrane int must be in {sorted(_VALID_MEMBRANE_CODES)}, got {value!r}"
            )
        return value
    if isinstance(value, str):
        if value not in MEMBRANE_LABEL_TO_CODE:
            raise ValueError(
                "membrane str must be exactly one of "
                f"{list(MEMBRANE_LABEL_TO_CODE.keys())}, got {value!r}"
            )
        return MEMBRANE_LABEL_TO_CODE[value]
    raise TypeError(f"membrane must be int, str, or None, got {type(value).__name__}")


# -----------------------------------------------------------------------------
# Search API types (aligned with ref/routers: peptide routes on /proteins, embedding-search, bio-source)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingSearchHit:
    rank: int
    internal_id: int
    similarity: float


@dataclass(frozen=True)
class EmbeddingSearchResult:
    query_length: int
    total: int
    results: List[EmbeddingSearchHit]


@dataclass(frozen=True)
class PeptidePropRow:
    internal_id: int
    seq_length: int
    net_charge: Optional[float] = None
    hydrophobicity: Optional[float] = None
    pis: Optional[float] = None
    localization: Optional[int] = None
    membrane: Optional[int] = None
    camp_score: Optional[float] = None


@dataclass(frozen=True)
class PaginatedPeptidePropResult:
    total: int
    page: int
    page_size: int
    has_next: bool
    data: List[PeptidePropRow]


@dataclass(frozen=True)
class PeptideSourceLinkRow:
    internal_id: int
    source_id: int
    genome_id: Optional[str] = None
    biome: Optional[str] = None
    habitat: Optional[str] = None
    source: Optional[str] = None
    latest_release: Optional[str] = None
    domain: Optional[str] = None
    phylum: Optional[str] = None
    taxonomy_class: Optional[str] = None
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None


@dataclass(frozen=True)
class PaginatedPeptideSourceLinkResult:
    total: int
    page: int
    page_size: int
    has_next: bool
    data: List[PeptideSourceLinkRow]


@dataclass(frozen=True)
class BioSourceRow:
    source_id: int
    genome_id: Optional[str] = None
    biome: Optional[str] = None
    habitat: Optional[str] = None
    source: Optional[str] = None
    latest_release: Optional[str] = None
    domain: Optional[str] = None
    phylum: Optional[str] = None
    taxonomy_class: Optional[str] = None
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None


@dataclass(frozen=True)
class PaginatedBioSourceResult:
    total: int
    page: int
    page_size: int
    has_next: bool
    data: List[BioSourceRow]


@dataclass(frozen=True)
class TaxonomyNode:
    value: str
    count: int


@dataclass(frozen=True)
class TaxonomyDrilldownResult:
    level: str
    parent_filters: Dict[str, Any]
    data: List[TaxonomyNode]


@dataclass(frozen=True)
class PeptideMainRow:
    internal_id: int
    accession_id: str
    seq_length: int
    family_50aai: Optional[int] = None
    sequence: Optional[str] = None


@dataclass(frozen=True)
class PeptidesByInternalIdResult:
    data: List[PeptideMainRow]


@dataclass(frozen=True)
class PeptidePropertiesRecord:
    internal_id: int
    seq_length: int
    camp_score: float
    tiny: int
    small: int
    aliphatic: int
    aromatic: int
    non_polar: int
    polar: int
    charged: int
    basic: int
    acidic: int
    aliphatic_index: float
    boman_index: float
    net_charge: float
    hydrophobicity: float
    instability: float
    pis: float
    hmoment_alpha: float
    hmoment_beta: float
    localization: int
    membrane: int
    disorder: Optional[str] = None
    dssp3: Optional[str] = None
    dssp8: Optional[str] = None


@dataclass(frozen=True)
class PeptideSourceRecord:
    source_id: int
    genome_id: Optional[str] = None
    biome: Optional[str] = None
    habitat: Optional[str] = None
    source: Optional[str] = None
    latest_release: Optional[str] = None
    domain: Optional[str] = None
    phylum: Optional[str] = None
    taxonomy_class: Optional[str] = None
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    contig_id: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    strand: Optional[str] = None


@dataclass(frozen=True)
class PaginatedPeptideSourceResult:
    total: int
    page: int
    page_size: int
    has_next: bool
    data: List[PeptideSourceRecord]


@dataclass(frozen=True)
class PaginatedPeptideMainResult:
    total: int
    page: int
    page_size: int
    has_next: bool
    data: List[PeptideMainRow]


class SearchStrategyKind(str, Enum):
    """Named search entry points matching the FastAPI routers under ref/routers."""

    EMBEDDING_SIMILARITY = "embedding_similarity"
    PEPTIDE_PROP = "peptide_prop"
    PEPTIDE_SOURCE_LINK = "peptide_source_link"
    BIO_SOURCE = "bio_source"
    TAXONOMY = "taxonomy"
    PEPTIDE_BY_INTERNAL_ID = "peptide_by_internal_id"
    PEPTIDE_PROPERTIES = "peptide_properties"
    PEPTIDE_SOURCES = "peptide_sources"
    PEPTIDE_BY_FAMILY = "peptide_by_family"


def embedding_search_similar(
    sequence: str,
    k: int = 10,
    *,
    session: Optional[requests.Session] = None,
    timeout: Optional[Union[float, Tuple[float, float]]] = None,
) -> EmbeddingSearchResult:
    """POST /embedding-search/search — FAISS similarity by ProtT5 embedding (ref embedding_search)."""
    if len(sequence) < 5:
        raise ValueError("sequence must be at least 5 amino acids")

    session = session or _get_session()
    url = _url("/embedding-search/search")
    payload = {"sequence": sequence, "k": k}
    read_timeout = timeout if timeout is not None else EMBEDDING_SEARCH_TIMEOUT
    logging.debug("POST %s body keys=%s", url, list(payload.keys()))
    try:
        r = session.post(
            url,
            json=payload,
            headers={**DEFAULT_HEADERS, "accept": "application/json"},
            timeout=read_timeout,
        )
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout embedding search (len=%s)", len(sequence))
        raise
    except RequestException:
        logging.exception("Embedding search request failed")
        raise

    hits = [
        EmbeddingSearchHit(
            rank=h["rank"],
            internal_id=h["internal_id"],
            similarity=float(h["similarity"]),
        )
        for h in raw.get("results", [])
    ]
    return EmbeddingSearchResult(
        query_length=int(raw["query_length"]),
        total=int(raw["total"]),
        results=hits,
    )


def search_peptide_prop(
    length_bin: List[int],
    *,
    net_charge_min: Optional[float] = None,
    net_charge_max: Optional[float] = None,
    hydrophobicity_min: Optional[float] = None,
    hydrophobicity_max: Optional[float] = None,
    pis_min: Optional[float] = None,
    pis_max: Optional[float] = None,
    localization: Optional[Union[int, str]] = None,
    membrane: Optional[Union[int, str]] = None,
    page: int = 1,
    page_size: int = 20,
    session: Optional[requests.Session] = None,
) -> PaginatedPeptidePropResult:
    """GET /proteins/search-peptide-prop — property-first scan (ref peptide router).

    ``localization`` and ``membrane`` may be API ints or exact labels from
    ``LOCALIZATION_LABEL_TO_CODE`` and ``MEMBRANE_LABEL_TO_CODE``; strings are
    sent to the API as integers.
    """
    session = session or _get_session()
    loc_api = _parse_localization_for_api(localization)
    mem_api = _parse_membrane_for_api(membrane)
    params: Dict[str, Any] = _prune_params(
        {
            "net_charge_min": net_charge_min,
            "net_charge_max": net_charge_max,
            "hydrophobicity_min": hydrophobicity_min,
            "hydrophobicity_max": hydrophobicity_max,
            "pis_min": pis_min,
            "pis_max": pis_max,
            "localization": loc_api,
            "membrane": mem_api,
            "page": page,
            "page_size": page_size,
        }
    )
    params["length_bin"] = length_bin
    url = _url("/proteins/search-peptide-prop")
    logging.debug("GET %s params=%s", url, params)
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout search_peptide_prop")
        raise
    except RequestException:
        logging.exception("search_peptide_prop failed")
        raise

    rows = [
        PeptidePropRow(
            internal_id=int(x["internal_id"]),
            seq_length=int(x["seq_length"]),
            net_charge=x.get("net_charge"),
            hydrophobicity=x.get("hydrophobicity"),
            pis=x.get("pis"),
            localization=x.get("localization"),
            membrane=x.get("membrane"),
            camp_score=x.get("camp_score"),
        )
        for x in raw.get("data", [])
    ]
    return PaginatedPeptidePropResult(
        total=int(raw["total"]),
        page=int(raw["page"]),
        page_size=int(raw["page_size"]),
        has_next=bool(raw["has_next"]),
        data=rows,
    )


def search_peptide_source_link(
    length_bin: List[int],
    *,
    biome: Optional[str] = None,
    habitat: Optional[str] = None,
    species: Optional[str] = None,
    genus: Optional[str] = None,
    family: Optional[str] = None,
    order: Optional[str] = None,
    taxonomy_class: Optional[str] = None,
    phylum: Optional[str] = None,
    domain: Optional[str] = None,
    genome_id: Optional[str] = None,
    source: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    session: Optional[requests.Session] = None,
) -> PaginatedPeptideSourceLinkResult:
    """GET /proteins/peptide-source-link-search — peptides by bio-source filters (ref peptide)."""
    session = session or _get_session()
    params: Dict[str, Any] = _prune_params(
        {
            "biome": biome,
            "habitat": habitat,
            "species": species,
            "genus": genus,
            "family": family,
            "order": order,
            "class": taxonomy_class,
            "phylum": phylum,
            "domain": domain,
            "genome_id": genome_id,
            "source": source,
            "page": page,
            "page_size": page_size,
        }
    )
    params["length_bin"] = length_bin
    url = _url("/proteins/peptide-source-link-search")
    logging.debug("GET %s params=%s", url, params)
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout search_peptide_source_link")
        raise
    except RequestException:
        logging.exception("search_peptide_source_link failed")
        raise

    rows = []
    for x in raw.get("data", []):
        rows.append(
            PeptideSourceLinkRow(
                internal_id=int(x["internal_id"]),
                source_id=int(x["source_id"]),
                genome_id=x.get("genome_id"),
                biome=x.get("biome"),
                habitat=x.get("habitat"),
                source=x.get("source"),
                latest_release=x.get("latest_release"),
                domain=x.get("domain"),
                phylum=x.get("phylum"),
                taxonomy_class=x.get("class"),
                order=x.get("order"),
                family=x.get("family"),
                genus=x.get("genus"),
                species=x.get("species"),
            )
        )
    return PaginatedPeptideSourceLinkResult(
        total=int(raw["total"]),
        page=int(raw["page"]),
        page_size=int(raw["page_size"]),
        has_next=bool(raw["has_next"]),
        data=rows,
    )


def search_bio_source(
    *,
    q: Optional[str] = None,
    field: Optional[str] = None,
    biome: Optional[str] = None,
    habitat: Optional[str] = None,
    species: Optional[str] = None,
    genus: Optional[str] = None,
    family: Optional[str] = None,
    order: Optional[str] = None,
    taxonomy_class: Optional[str] = None,
    phylum: Optional[str] = None,
    domain: Optional[str] = None,
    source: Optional[str] = None,
    genome_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    session: Optional[requests.Session] = None,
) -> PaginatedBioSourceResult:
    """GET /bio-source/search (ref bio_source)."""
    session = session or _get_session()
    params = _prune_params(
        {
            "q": q,
            "field": field,
            "biome": biome,
            "habitat": habitat,
            "species": species,
            "genus": genus,
            "family": family,
            "order": order,
            "class": taxonomy_class,
            "phylum": phylum,
            "domain": domain,
            "source": source,
            "genome_id": genome_id,
            "page": page,
            "page_size": page_size,
        }
    )
    url = _url("/bio-source/search")
    logging.debug("GET %s params=%s", url, params)
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout search_bio_source")
        raise
    except RequestException:
        logging.exception("search_bio_source failed")
        raise

    rows = []
    for x in raw.get("data", []):
        rows.append(
            BioSourceRow(
                source_id=int(x["source_id"]),
                genome_id=x.get("genome_id"),
                biome=x.get("biome"),
                habitat=x.get("habitat"),
                source=x.get("source"),
                latest_release=x.get("latest_release"),
                domain=x.get("domain"),
                phylum=x.get("phylum"),
                taxonomy_class=x.get("class"),
                order=x.get("order"),
                family=x.get("family"),
                genus=x.get("genus"),
                species=x.get("species"),
            )
        )
    return PaginatedBioSourceResult(
        total=int(raw["total"]),
        page=int(raw["page"]),
        page_size=int(raw["page_size"]),
        has_next=bool(raw["has_next"]),
        data=rows,
    )


def bio_source_taxonomy(
    level: str,
    *,
    domain: Optional[str] = None,
    phylum: Optional[str] = None,
    taxonomy_class: Optional[str] = None,
    order: Optional[str] = None,
    family: Optional[str] = None,
    genus: Optional[str] = None,
    limit: int = 200,
    session: Optional[requests.Session] = None,
) -> TaxonomyDrilldownResult:
    """GET /bio-source/taxonomy — drill-down counts (ref bio_source)."""
    session = session or _get_session()
    params = _prune_params(
        {
            "level": level,
            "domain": domain,
            "phylum": phylum,
            "class": taxonomy_class,
            "order": order,
            "family": family,
            "genus": genus,
            "limit": limit,
        }
    )
    url = _url("/bio-source/taxonomy")
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout bio_source_taxonomy")
        raise
    except RequestException:
        logging.exception("bio_source_taxonomy failed")
        raise

    nodes = [
        TaxonomyNode(value=str(n["value"]), count=int(n["count"]))
        for n in raw.get("data", [])
    ]
    return TaxonomyDrilldownResult(
        level=str(raw["level"]),
        parent_filters=dict(raw.get("parent_filters") or {}),
        data=nodes,
    )


def get_peptides_by_internal_id(
    internal_id: int,
    *,
    include_sequence: bool = True,
    session: Optional[requests.Session] = None,
) -> PeptidesByInternalIdResult:
    """GET /proteins/by-internal-id/{internal_id} (ref peptide)."""
    session = session or _get_session()
    url = _url(f"/proteins/by-internal-id/{internal_id}")
    params = {"include_sequence": "true" if include_sequence else "false"}
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout get_peptides_by_internal_id %s", internal_id)
        raise
    except RequestException:
        logging.exception("get_peptides_by_internal_id failed")
        raise

    rows = [
        PeptideMainRow(
            internal_id=int(x["internal_id"]),
            accession_id=str(x["accession_id"]),
            seq_length=int(x["seq_length"]),
            family_50aai=x.get("family_50aai"),
            sequence=x.get("sequence"),
        )
        for x in raw.get("data", [])
    ]
    return PeptidesByInternalIdResult(data=rows)


def get_peptide_properties_by_internal_id(
    internal_id: int,
    *,
    seq_length: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> PeptidePropertiesRecord:
    """GET /proteins/by-internal-id/{internal_id}/properties (ref peptide)."""
    session = session or _get_session()
    url = _url(f"/proteins/by-internal-id/{internal_id}/properties")
    params = _prune_params({"seq_length": seq_length})
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout get_peptide_properties_by_internal_id %s", internal_id)
        raise
    except RequestException:
        logging.exception("get_peptide_properties_by_internal_id failed")
        raise

    return PeptidePropertiesRecord(**raw)


def list_peptide_sources_by_internal_id(
    internal_id: int,
    *,
    seq_length: Optional[int] = None,
    page: int = 1,
    page_size: int = 20,
    session: Optional[requests.Session] = None,
) -> PaginatedPeptideSourceResult:
    """GET /proteins/by-internal-id/{internal_id}/sources (ref peptide)."""
    session = session or _get_session()
    url = _url(f"/proteins/by-internal-id/{internal_id}/sources")
    params = _prune_params(
        {"seq_length": seq_length, "page": page, "page_size": page_size}
    )
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout list_peptide_sources_by_internal_id %s", internal_id)
        raise
    except RequestException:
        logging.exception("list_peptide_sources_by_internal_id failed")
        raise

    rows = []
    for x in raw.get("data", []):
        rows.append(
            PeptideSourceRecord(
                source_id=int(x["source_id"]),
                genome_id=x.get("genome_id"),
                biome=x.get("biome"),
                habitat=x.get("habitat"),
                source=x.get("source"),
                latest_release=x.get("latest_release"),
                domain=x.get("domain"),
                phylum=x.get("phylum"),
                taxonomy_class=x.get("class"),
                order=x.get("order"),
                family=x.get("family"),
                genus=x.get("genus"),
                species=x.get("species"),
                contig_id=x.get("contig_id"),
                start=x.get("start"),
                end=x.get("end"),
                strand=x.get("strand"),
            )
        )
    return PaginatedPeptideSourceResult(
        total=int(raw["total"]),
        page=int(raw["page"]),
        page_size=int(raw["page_size"]),
        has_next=bool(raw["has_next"]),
        data=rows,
    )


def get_peptides_by_family_50aai(
    family_50aai: int,
    *,
    include_sequence: bool = False,
    length_bin: Optional[List[int]] = None,
    page: int = 1,
    page_size: int = 20,
    session: Optional[requests.Session] = None,
) -> PaginatedPeptideMainResult:
    """GET /proteins/by-family-50aai/{family_50aai} (ref peptide)."""
    session = session or _get_session()
    url = _url(f"/proteins/by-family-50aai/{family_50aai}")
    params: Dict[str, Any] = _prune_params(
        {
            "include_sequence": "true" if include_sequence else "false",
            "page": page,
            "page_size": page_size,
        }
    )
    if length_bin:
        params["length_bin"] = length_bin
    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except Timeout:
        logging.error("Timeout get_peptides_by_family_50aai %s", family_50aai)
        raise
    except RequestException:
        logging.exception("get_peptides_by_family_50aai failed")
        raise

    rows = [
        PeptideMainRow(
            internal_id=int(x["internal_id"]),
            accession_id=str(x["accession_id"]),
            seq_length=int(x["seq_length"]),
            family_50aai=x.get("family_50aai"),
            sequence=x.get("sequence"),
        )
        for x in raw.get("data", [])
    ]
    return PaginatedPeptideMainResult(
        total=int(raw["total"]),
        page=int(raw["page"]),
        page_size=int(raw["page_size"]),
        has_next=bool(raw["has_next"]),
        data=rows,
    )


def run_search_strategies(
    steps: Sequence[Union[SearchStrategyKind, str]],
    kwargs_by_step: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    *,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Run multiple search strategies in order and return a dict of results keyed by strategy value.

    Pass parameters for each step via ``kwargs_by_step``, e.g.::

        run_search_strategies(
            [SearchStrategyKind.EMBEDDING_SIMILARITY, SearchStrategyKind.PEPTIDE_PROP],
            {
                "embedding_similarity": {"sequence": "ACDEFGHIKLMNPQR", "k": 5},
                "peptide_prop": {"length_bin": [10], "membrane": 1},
            },
        )
    """
    session = session or _get_session()
    kw = dict(kwargs_by_step or {})
    out: Dict[str, Any] = {}

    for step in steps:
        kind = (
            step
            if isinstance(step, SearchStrategyKind)
            else SearchStrategyKind(str(step))
        )
        key = kind.value
        args = dict(kw.get(key, {}))

        if kind is SearchStrategyKind.EMBEDDING_SIMILARITY:
            out[key] = embedding_search_similar(session=session, **args)
        elif kind is SearchStrategyKind.PEPTIDE_PROP:
            out[key] = search_peptide_prop(session=session, **args)
        elif kind is SearchStrategyKind.PEPTIDE_SOURCE_LINK:
            out[key] = search_peptide_source_link(session=session, **args)
        elif kind is SearchStrategyKind.BIO_SOURCE:
            out[key] = search_bio_source(session=session, **args)
        elif kind is SearchStrategyKind.TAXONOMY:
            out[key] = bio_source_taxonomy(session=session, **args)
        elif kind is SearchStrategyKind.PEPTIDE_BY_INTERNAL_ID:
            out[key] = get_peptides_by_internal_id(session=session, **args)
        elif kind is SearchStrategyKind.PEPTIDE_PROPERTIES:
            out[key] = get_peptide_properties_by_internal_id(session=session, **args)
        elif kind is SearchStrategyKind.PEPTIDE_SOURCES:
            out[key] = list_peptide_sources_by_internal_id(session=session, **args)
        elif kind is SearchStrategyKind.PEPTIDE_BY_FAMILY:
            out[key] = get_peptides_by_family_50aai(session=session, **args)
        else:
            raise ValueError(f"Unknown search strategy: {step!r}")

    return out


class MiProbeSearchClient:
    """Thin facade over ref-aligned search endpoints (same session for all calls)."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or _get_session()

    def embedding_similarity(
        self,
        sequence: str,
        k: int = 10,
        *,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> EmbeddingSearchResult:
        return embedding_search_similar(
            sequence, k, session=self.session, timeout=timeout
        )

    def peptide_prop(
        self, length_bin: List[int], **kwargs: Any
    ) -> PaginatedPeptidePropResult:
        return search_peptide_prop(length_bin, session=self.session, **kwargs)

    def peptide_source_link(
        self, length_bin: List[int], **kwargs: Any
    ) -> PaginatedPeptideSourceLinkResult:
        return search_peptide_source_link(length_bin, session=self.session, **kwargs)

    def bio_source(self, **kwargs: Any) -> PaginatedBioSourceResult:
        return search_bio_source(session=self.session, **kwargs)

    def taxonomy(self, level: str, **kwargs: Any) -> TaxonomyDrilldownResult:
        return bio_source_taxonomy(level, session=self.session, **kwargs)

    def peptides_by_internal_id(
        self, internal_id: int, **kwargs: Any
    ) -> PeptidesByInternalIdResult:
        return get_peptides_by_internal_id(internal_id, session=self.session, **kwargs)

    def peptide_properties(
        self, internal_id: int, **kwargs: Any
    ) -> PeptidePropertiesRecord:
        return get_peptide_properties_by_internal_id(
            internal_id, session=self.session, **kwargs
        )

    def peptide_sources(
        self, internal_id: int, **kwargs: Any
    ) -> PaginatedPeptideSourceResult:
        return list_peptide_sources_by_internal_id(
            internal_id, session=self.session, **kwargs
        )

    def peptides_by_family(
        self, family_50aai: int, **kwargs: Any
    ) -> PaginatedPeptideMainResult:
        return get_peptides_by_family_50aai(
            family_50aai, session=self.session, **kwargs
        )

    def run_strategies(
        self,
        steps: Sequence[Union[SearchStrategyKind, str]],
        kwargs_by_step: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return run_search_strategies(steps, kwargs_by_step, session=self.session)


# -----------------------------------------------------------------------------
# Core Functions (precomputed embedding API — ref/routers/embedding.py)
# -----------------------------------------------------------------------------

_EMBEDDING_MODEL_ALIASES = {
    "t5": "prott5",
    "prot_t5": "prott5",
    "prott5": "prott5",
    "esm2": "esm2",
}


def _normalize_embedding_model_type(name: str) -> str:
    key = (name or "prott5").strip().lower()
    return _EMBEDDING_MODEL_ALIASES.get(key, key)


def get_embedding_by_id(
    pid: str,
    embedding_model: str = "prott5",
    session: Optional[requests.Session] = None,
) -> np.ndarray:
    """Return the peptide embedding as a numpy.ndarray of dtype float32.

    Calls ``GET {BASE_URL}/embedding`` with ``accession_id`` and ``model_type``
    (``prott5`` or ``esm2``), matching ``ref/routers/embedding.py``.

    Parameters
    ----------
    pid : str
        Accession id (e.g. ``ORF050.00000001``), same as HDF5 dataset key on server.
    embedding_model : str, default ``"prott5"``
        Embedding backend: ``prott5`` or ``esm2``. Legacy alias ``t5`` maps to ``prott5``.
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
    model_type = _normalize_embedding_model_type(embedding_model)
    params = {
        "accession_id": pid.strip(),
        "model_type": model_type,
    }
    headers = {"accept": "application/json"}

    try:
        logging.debug("Requesting embedding: %s params=%s", url, params)
        r = session.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        response_data = r.json()
        if "embedding" not in response_data:
            raise ValueError("Invalid response format: missing 'embedding' field")

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
    embedding_model: str = "prott5",
    session: Optional[requests.Session] = None,
) -> np.ndarray:
    """Fetch embeddings for multiple peptide IDs and return stacked array.

    Parameters
    ----------
    ids : List[str]
        List of MiProbe identifiers
    embedding_model : str, default ``"prott5"``
        Name of the embedding model (``prott5`` or ``esm2``; ``t5`` aliases to ``prott5``).
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
    Utility class for downloading peptide embeddings via the precomputed embedding API.

    Note: This class is maintained for backward compatibility.
    Consider using the functional API instead.
    """

    def __init__(
        self,
        embedding_model: str = "prott5",
        session: Optional[requests.Session] = None,
    ):
        self.embedding_model = embedding_model
        self.session = session

    def get_embedding_by_id(self, pid: str) -> np.ndarray:
        """Return the peptide embedding as a numpy.ndarray of dtype float32."""
        return get_embedding_by_id(pid, self.embedding_model, self.session)

    def batch_fetch_embeddings(self, ids: List[str]) -> np.ndarray:
        """Fetch embeddings for multiple peptide IDs and return stacked array."""
        return batch_fetch_embeddings(ids, self.embedding_model, self.session)
