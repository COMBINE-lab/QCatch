import base64
import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests
from anndata import AnnData

logger = logging.getLogger(__name__)

# Define the standard snake_case columns - single source of truth
STANDARD_COLUMNS: list[str] = [
    "barcodes",
    "corrected_reads",
    "mapped_reads",
    "deduplicated_reads",
    "mapping_rate",
    "dedup_rate",
    "mean_by_max",
    "num_genes_expressed",
    "num_genes_over_mean",
]

# Only need this mapping if input is in CamelCase
CAMEL_TO_SNAKE_MAPPING = {
    "barcodes": "barcodes",  # stays the same
    "CorrectedReads": "corrected_reads",
    "MappedReads": "mapped_reads",
    "DeduplicatedReads": "deduplicated_reads",
    "MappingRate": "mapping_rate",
    "DedupRate": "dedup_rate",
    "MeanByMax": "mean_by_max",
    "NumGenesExpressed": "num_genes_expressed",
    "NumGenesOverMean": "num_genes_over_mean",
}


def standardize_feature_dump_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize feature dump columns to snake_case format.

    If columns are already in snake_case, validates them.
    If columns are in CamelCase, converts them to snake_case.
    Allows for additional columns beyond the standard ones.

    Parameters
    ----------
    df
        Input DataFrame containing feature dump columns.

    Returns
    -------
    DataFrame with standardized snake_case columns.
    """
    # Check if already in standard snake_case format (allow extra columns)
    # NOTE: deprecated the 'num_expressed' column conversion. This input will not be supported in the future.
    if "num_expressed" in df.columns:
        df = df.rename(columns={"num_expressed": "num_genes_expressed"})

    if set(STANDARD_COLUMNS).issubset(df.columns):
        return df

    # If the DataFrame columns match the keys in the CamelCase mapping, convert them to snake_case (allowing extra columns)
    if set(CAMEL_TO_SNAKE_MAPPING.keys()).issubset(df.columns):
        renamed_df = df.rename(columns=CAMEL_TO_SNAKE_MAPPING)
        return renamed_df[STANDARD_COLUMNS]

    # If neither format matches, raise error
    raise ValueError(
        "Input columns must match either standard snake_case or expected CamelCase format. "
        f"Expected snake_case columns: {STANDARD_COLUMNS}"
    )


def load_json_txt_file(parent_dir: str) -> tuple[dict, dict, pd.DataFrame]:
    """
    Load quant.json, generate_permit_list.json, and featureDump.txt from the given directory.

    Parameters
    ----------
    parent_dir
        Path to the directory containing the input files.

    Returns
    -------
    Tuple containing:
        - Dictionary of quant.json data.
        - Dictionary of permit_list.json data.
        - DataFrame of feature dump.
    """
    quant_json_data_path = Path(os.path.join(parent_dir, "quant.json"))
    permit_list_path = Path(os.path.join(parent_dir, "generate_permit_list.json"))
    feature_dump_path = Path(os.path.join(parent_dir, "featureDump.txt"))

    # Check if quant.json exists
    if not quant_json_data_path.exists():
        logger.error(f"❌ Error: Missing required file: '{quant_json_data_path}'")
        quant_json_data = {}
    else:
        with open(quant_json_data_path) as f:
            quant_json_data = json.load(f)

    # Check if generate_permit_list.json exists
    if not permit_list_path.exists():
        permit_list_json_data = {}
    else:
        with open(permit_list_path) as f:
            permit_list_json_data = json.load(f)

    # Check if feature_dump.txt exists
    if not feature_dump_path.exists():
        logger.error(f"❌ Error: Missing required file: '{feature_dump_path}'")
        raise ValueError(f"Missing required file: '{feature_dump_path}'")
    else:
        feature_dump_data = pd.read_csv(feature_dump_path, sep="\t")
        feature_dump_data.columns = STANDARD_COLUMNS

    return quant_json_data, permit_list_json_data, feature_dump_data


# from https://ga4gh.github.io/refget/seqcols/
def canonical_str(item: [list, dict]) -> bytes:
    """Convert a list or dict into a canonicalized UTF-8 encoded bytestring."""
    return json.dumps(item, separators=(",", ":"), ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf8")


def sha512t24u_digest(seq: bytes) -> str:
    """
    Compute the GA4GH digest function.

    Parameters
    ----------
    seq
        Input bytes sequence.

    Returns
    -------
    Truncated base64 URL-safe digest.
    """
    offset = 24
    digest = hashlib.sha512(seq).digest()
    tdigest_b64us = base64.urlsafe_b64encode(digest[:offset])
    return tdigest_b64us.decode("ascii")


def get_name_digest(item: list) -> str:
    """Compute the name digest for a given list."""
    return sha512t24u_digest(canonical_str(item))


def get_name_mapping_file_from_registry(seqcol_digest, output_dir):
    """
    Fetch a gene ID-to-name mapping file from a remote registry.

    Parameters
    ----------
    seqcol_digest
        Digest string for the sequence collection.
    output_dir
        Directory to save the downloaded file.

    Returns
    -------
    Path or None
        Path to the downloaded file if successful, otherwise None.
    """
    output_file = output_dir / f"{seqcol_digest}.tsv"
    REGISTRY_URL = "https://raw.githubusercontent.com/COMBINE-lab/QCatch-resources/refs/heads/main/resources/registries/id2name.json"
    r = requests.get(REGISTRY_URL)
    if r.ok:
        reg = r.json()
        if seqcol_digest in reg:
            file_url = reg[seqcol_digest]["url"]
            logger.info(f"✅ found entry for {seqcol_digest} in registry; fetching file from {file_url}")
            r = requests.get(file_url, stream=True)
            with open(output_file, mode="wb") as file:
                for chunk in r.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)

            if not output_file.exists():
                logger.error("❌ downloaded file not found")
                return None
            return output_file
        else:
            return None


def add_gene_symbol(adata: AnnData, gene_id2name_file: Path | None, output_dir: Path) -> AnnData:
    """
    Add gene symbols to an AnnData object based on a gene ID-to-name mapping.

    Parameters
    ----------
    adata
        Input AnnData object.
    gene_id2name_file
        Path to the gene ID-to-name mapping file. If None, attempts to fetch from the registry.
    output_dir
        Directory to save any downloaded mapping files.

    Returns
    -------
    Updated AnnData object with gene symbols added.
    """
    if adata.var.index.names == ["gene_ids"]:
        # from mtx data
        all_gene_ids = adata.var.index
    else:
        # from h5ad data
        if "gene_id" in adata.var:
            all_gene_ids = adata.var["gene_id"]
        elif "gene_ids" in adata.var:
            # from original simpleaf mtx data
            all_gene_ids = adata.var["gene_ids"]
        else:
            logger.error("❌ Error: Neither 'gene_id' nor 'gene_ids' found in adata.var columns; cannot add mapping")
            return adata
    # check the digest for this adata object
    all_gene_ids = pd.Series(all_gene_ids)
    seqcol_digest = get_name_digest(sorted(all_gene_ids.to_list()))
    logger.info(f"the seqcol digest for the sorted gene ids is : {seqcol_digest}")
    # What we will try to get the mapping
    #
    # 1) if the user provided nothing, check the registry and see if
    # we can fetch an associated file. If so, fetch and use it
    #
    # 2) if the user provided a file directly, make sure that
    # the digest of the file matches what is expected and then use the mapping.
    gene_id2name_path = None

    if gene_id2name_file is None:
        gene_id2name_path = get_name_mapping_file_from_registry(seqcol_digest, output_dir)
        if gene_id2name_path is None:
            logger.warning("Failed to properly obtain gene id-to-name mapping; will not add mapping")
            return adata
    elif gene_id2name_file.exists() and gene_id2name_file.is_file():
        gene_id2name_path = gene_id2name_file
    else:
        logger.warning(
            "If gene id-to-name mapping is provided, it should be a file, but a directory was provided; will not add mapping"
        )
        return adata
    # add the gene symbol, based on the gene id to symbol mapping
    gene_id_to_symbol = pd.read_csv(gene_id2name_path, sep="\t", header=None, names=["gene_id", "gene_name"])
    # Identify missing gene symbols

    missing_symbols_count = gene_id_to_symbol["gene_name"].isna().sum()

    if missing_symbols_count > 0:
        logger.info(f"Number of gene IDs with missing gene_name/symbols: {missing_symbols_count}")
        # Replace NaN values in 'gene_symbol' with the corresponding 'gene_id'
        gene_id_to_symbol["gene_name"].fillna(gene_id_to_symbol["gene_id"], inplace=True)
        logger.info("Filled missing symbols with gene_id.")
    # Create a mapping dictionary
    id_to_symbol_dict = pd.Series(gene_id_to_symbol["gene_name"].values, index=gene_id_to_symbol["gene_id"]).to_dict()
    # Initialize an empty list to hold the reordered symbols
    reordered_symbols = []
    # Iterate through 'all_gene_ids' and fetch corresponding symbols
    for gene_id in all_gene_ids:
        symbol = id_to_symbol_dict.get(gene_id)
        reordered_symbols.append(symbol)
    #  Integrate the Reordered Mapping into AnnData
    # Assign gene symbols to AnnData's .var attribute

    adata.var["gene_symbol"] = reordered_symbols
    # (Optional) Replace var_names with gene symbols
    # This can make plots and analyses more interpretable
    adata.var_names = adata.var["gene_symbol"].astype(str)
    # Ensure uniqueness of var_names after replacement
    adata.var_names_make_unique(join="-")

    return adata
