import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scanpy as sc
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
            logger.info(f"Found entry for {seqcol_digest} in registry; fetching file from {file_url}")

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
    logger.info("🥨 Trying to add gene symbols based on the gene id to name mapping.")
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
    logger.info("✅ Added the gene symbol!")
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


def parse_user_valid_cell_list(valid_cell_list: str | Path) -> list[str]:
    """
    Parse a user-provided valid cell barcode list file.

    The CLI contract (see `main.py`) is:
    - TSV containing **one** column of barcodes
    - **no header**

    In practice, we accept either:
    - one barcode per line, or
    - a TSV where the barcode is the first field on each line

    Returns a de-duplicated list (order-preserving).
    """
    path = Path(valid_cell_list)
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError(f"--valid_cell_list must be a file path; got: {path}")

    barcodes: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            fields = [x.strip() for x in line.split("\t")]
            bc = fields[0]

            # Be forgiving if someone accidentally includes a header.
            if line_no == 1 and bc.lower() in {"barcode", "barcodes"}:
                continue

            # Enforce the "one column" contract: if additional columns exist and are non-empty, fail fast.
            if len(fields) > 1 and any(x for x in fields[1:]):
                raise argparse.ArgumentTypeError(
                    f"--valid_cell_list must be a 1-column TSV (barcode only). "
                    f"Found extra columns on line {line_no}: {line}"
                )

            if not bc:
                continue

            if bc not in seen:
                seen.add(bc)
                barcodes.append(bc)

    if not barcodes:
        raise argparse.ArgumentTypeError(f"--valid_cell_list file is empty (or only comments/blank lines): {path}")

    return barcodes


def compute_mito_percent(adata: AnnData) -> None:
    """Compute mitochondrial percentage in-place."""
    adata.var["mt"] = adata.var["gene_symbol"].str.startswith("MT-") | adata.var["gene_symbol"].str.startswith("mt-")
    if adata.var["mt"].sum() > 0:
        # Compute QC metrics
        obs_qc, _ = sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=False, percent_top=None, log1p=False)
        adata.obs["pct_counts_mt"] = obs_qc["pct_counts_mt"]
        logger.info("🥖 Mitochondrial percentage computed and added to adata.obs.")
    else:
        logger.warning("No mitochondrial genes found.")


def remove_doublets(adata, valid_bcs) -> list[str]:
    """
    Detect doublets using Scrublet and return barcodes of retained singlets.

    This function runs Scrublet on the subset of cells defined by `valid_bcs`, writes Scrublet
    outputs back to the full `adata.obs` in-place (`doublet_score`, `predicted_doublet`), then
    returns the subset barcodes that are not predicted doublets.

    Parameters
    ----------
    adata
        AnnData object. Must contain `adata.obs["barcodes"]`.
    valid_bcs
        Iterable of barcodes to evaluate with Scrublet.

    Returns
    -------
    list[str]
        Barcodes of retained singlets (i.e., not predicted doublets).
    """
    logger.info("👀 Detecting doublets (use `Scrublet` tool)...")
    # work on a copy to avoid view-related issues
    current_retained_cells = adata[adata.obs["barcodes"].isin(valid_bcs)].copy()

    n_found = len(current_retained_cells)
    logger.info(f"🔬 Running Scrublet on {n_found} cells...")
    sc.pp.scrublet(current_retained_cells)

    # Check if Scrublet filtered any cells
    n_after_scrublet = len(current_retained_cells)
    if n_after_scrublet < n_found:
        logger.warning(
            f"⚠️ Scrublet filtered out {n_found - n_after_scrublet} cells (had {n_found}, now {n_after_scrublet})"
        )

    # fill with NA for all cells
    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = pd.Series(
        pd.array([pd.NA] * adata.n_obs, dtype="boolean"),
        index=adata.obs_names,
    )

    # add doublet score and predicted doublet to cells in current_retained_cells
    idx = current_retained_cells.obs_names
    adata.obs.loc[idx, "doublet_score"] = current_retained_cells.obs["doublet_score"].to_numpy()
    adata.obs.loc[idx, "predicted_doublet"] = current_retained_cells.obs["predicted_doublet"].to_numpy()

    keep = ~current_retained_cells.obs["predicted_doublet"].astype(bool)
    valid_bcs = (
        current_retained_cells.obs.loc[keep, "barcodes"].astype(str).str.strip().tolist()
        if "barcodes" in current_retained_cells.obs
        else current_retained_cells.obs_names[keep].astype(str).tolist()
    )
    logger.info(f"✅ Retained cells after removing doublets: {len(valid_bcs)}")

    return valid_bcs


def save_results(args, version, intermediate_result, valid_bcs):
    """Save the cell calling results for h5ad or mtx directory."""
    if intermediate_result is not None:
        converted_filtered_bcs, non_ambient_result = intermediate_result
    else:
        converted_filtered_bcs, non_ambient_result = None, None

    # add qcatch version
    qcatch_log = {
        "version": version,
    }
    output_dir = args.output
    # Save the cell calling result

    # Always compute mitochondrial percentage (in-place) when possible
    if "gene_symbol" in args.input.mtx_data.var.columns:
        compute_mito_percent(args.input.mtx_data)
    # check if any result columns already exist
    existing_cols = {
        "initial_filtered_cell",
        "potential_non_ambient_cell",
        "non_ambient_pvalue",
        "is_retained_cells",
    }
    if existing_cols.intersection(args.input.mtx_data.obs.columns):
        logger.warning(
            "⚠️ Cell calling result columns already exist in the h5ad file will be removed before being overwritten with new QCatch analyis."
        )
        # remove the existing columns
        args.input.mtx_data.obs.drop(columns=existing_cols.intersection(args.input.mtx_data.obs.columns), inplace=True)

    if not args.valid_cell_list:
        # Update the h5ad file with the final retain cells, contains original filtered cells and passed non-ambient cells
        args.input.mtx_data.obs["initial_filtered_cell"] = args.input.mtx_data.obs["barcodes"].isin(
            converted_filtered_bcs
        )
        # save the non-ambient cells, if available
        if non_ambient_result is not None:
            args.input.mtx_data.obs["potential_non_ambient_cell"] = args.input.mtx_data.obs["barcodes"].isin(
                non_ambient_result.eval_bcs
            )
            # Create a mapping from barcodes to p-values
            barcode_to_pval = dict(zip(non_ambient_result.eval_bcs, non_ambient_result.pvalues, strict=False))
            # Assign p-values only where 'is_nonambient' is True, otherwise fill with NaN
            args.input.mtx_data.obs["non_ambient_pvalue"] = (
                args.input.mtx_data.obs["barcodes"].map(barcode_to_pval).astype("float")
            )
        logger.info("🗂️ Add 'cell calling result' to the h5ad file, check the new added columns in adata.obs .")

    args.input.mtx_data.obs["is_retained_cells"] = args.input.mtx_data.obs["barcodes"].isin(valid_bcs)

    args.input.mtx_data.uns["qc_info"] = qcatch_log

    if "gene_symbol" in args.input.mtx_data.var.index.name:
        logger.info(
            "✅ The gene symbol is already in the adata.var.index.name, rename the 'gene_symbol' column in adata.var to 'gene_symbol_added'"
        )
        args.input.mtx_data.var.rename(columns={"gene_symbol": "gene_symbol_added"}, inplace=True)

    if args.input.is_h5ad and output_dir == args.input.dir:
        # Inplace overwrite: same location as original
        temp_file = os.path.join(output_dir, "quants.h5ad")
        args.input.mtx_data.write_h5ad(temp_file, compression="gzip")
        input_h5ad_file = args.input.file
        os.remove(input_h5ad_file)
        shutil.move(temp_file, input_h5ad_file)
        logger.info("📋 Overwrote the original h5ad file with the new cell calling result.")
    else:
        # Save to a new h5ad
        output_h5ad_file = os.path.join(output_dir, "quants.h5ad")
        args.input.mtx_data.write_h5ad(output_h5ad_file, compression="gzip")
        logger.info(f"📋 Saved h5ad file with new metadata to: {output_h5ad_file}")
        if not args.input.is_h5ad:
            logger.info(
                "Since the input is a mtx directory, the new quants.h5ad file only contains the metadata updated by QCatch. Other metadata are still in the original alevin output directory."
            )

    if args.save_filtered_h5ad:
        # filter the anndata, only keep the cells in valid_bcs
        filter_mtx_data = args.input.mtx_data[args.input.mtx_data.obs["is_retained_cells"].values, :].copy()
        # Save the filtered anndata to a new file
        filter_mtx_data_filename = os.path.join(output_dir, "filtered_quants.h5ad")
        filter_mtx_data.write_h5ad(filter_mtx_data_filename, compression="gzip")
        logger.info(f"📋 Saved the filtered h5ad file to {filter_mtx_data_filename}.")

    # # mtx dir input -->

    #     # Save the total retained cells to a txt file
    #     final_retained_cell_file = os.path.join(output_dir, "final_retained_cells.txt")
    #     with open(final_retained_cell_file, "w") as f:
    #         for bc in valid_bcs:
    #             f.write(f"{bc}\n")
    #     # Logging the cell calling result path
    #     logger.info(f"🗂️ Saved cell calling result and qcatch log file in the output directory: {output_dir}")
    #     # Save the qcatch log file. abou the version
    #     qcatch_log_file = os.path.join(output_dir, "qcatch_log.txt")
    #     with open(qcatch_log_file, "w") as f:
    #         for key, value in qcatch_log.items():
    #             f.write(f"{key}: {value}\n")
    #     logger.info(f"🗂️ Saved qcatch log file in the output directory: {output_dir}")
