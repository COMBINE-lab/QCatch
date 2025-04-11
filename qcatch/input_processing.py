import os
import pandas as pd
import json
import base64
import hashlib
import requests
from pyroe import load_fry
import logging
import scanpy as sc
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Define the standard snake_case columns - single source of truth
STANDARD_COLUMNS: List[str] = [
    'barcodes',
    'corrected_reads',
    'mapped_reads',
    'deduplicated_reads',
    'mapping_rate',
    'dedup_rate',
    'mean_by_max',
    'num_expressed',
    'num_genes_over_mean'
]

# Only need this mapping if input is in CamelCase
CAMEL_TO_SNAKE_MAPPING = {
    'barcodes': 'barcodes',  # stays the same
    'CorrectedReads': 'corrected_reads',
    'MappedReads': 'mapped_reads',
    'DeduplicatedReads': 'deduplicated_reads',
    'MappingRate': 'mapping_rate',
    'DedupRate': 'dedup_rate',
    'MeanByMax': 'mean_by_max',
    'NumGenesExpressed': 'num_expressed',
    'NumGenesOverMean': 'num_genes_over_mean'
}

def standardize_feature_dump_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize feature dump columns to snake_case format.
    If columns are already in snake_case, validates them.
    If columns are in CamelCase, converts them to snake_case.
    """
    # Check if already in standard snake_case format
    if set(df.columns) == set(STANDARD_COLUMNS):
        return df[STANDARD_COLUMNS]  # just ensure column order
    
    # If not snake_case, try converting from CamelCase
    if set(df.columns) == set(CAMEL_TO_SNAKE_MAPPING.keys()):
        return df.rename(columns=CAMEL_TO_SNAKE_MAPPING)[STANDARD_COLUMNS]
    
    # If neither format matches, raise error
    raise ValueError(
        "Input columns must match either standard snake_case or expected CamelCase format. "
        f"Expected snake_case columns: {STANDARD_COLUMNS}"
    )

def parse_quant_out_dir(quant_out_dir):
    """
    Detects the input format of the quantification output directory.
    return the loaded data
    """
    # check if input is a directory or a file
    if os.path.isfile(quant_out_dir):
        # it's a file and must be an H5AD file so we deal with that here
        is_h5ad = True
        h5ad_file_path = quant_out_dir
        logger.info("✅ Loading the data from h5ad file...")
        mtx_data = sc.read_h5ad(h5ad_file_path)
        quant_json_data, permit_list_json_data = json.loads(mtx_data.uns['quant_info']), json.loads(mtx_data.uns['gpl_info'])

        feature_dump_data = pd.DataFrame(mtx_data.obs)
        # Standardize feature dump columns to snake_case format
        standardize_feature_dump_columns(feature_dump_data)
        usa_mode = quant_json_data['usa_mode']

    else:
        used_simpleaf = None
        if os.path.exists(os.path.join(quant_out_dir, 'simpleaf_quant')):
            logger.info("✅ Detected: 'simpleaf' was used for the quantification result.")
            used_simpleaf = True
        elif os.path.exists(os.path.join(quant_out_dir, 'quant.json')):
            logger.info("✅ Detected: 'alevin-fry' was used for the quantification result.")
            used_simpleaf = False
        else:
            logger.warning(
                "⚠️ Unable to recognize the quantification directory. "
                "Ensure that the directory structure remains unchanged from the original output directory."
            )
        
        # -----------------------------------
        # Loads matrix data from the given quantification output directory.
        mtx_dir_path = os.path.join(quant_out_dir, "simpleaf_quant", "af_quant") if used_simpleaf else quant_out_dir

        if not os.path.exists(mtx_dir_path):
            logger.error(f"❌ Error: Expected matrix directory '{mtx_dir_path}' not found. Please check the input directory structure.")
            mtx_data = None
            
        is_h5ad = False
        # -----------------------------------
        # Check if quants.h5ad file exists in the parent directory
        h5ad_file_path = os.path.join(quant_out_dir, 'quants.h5ad')
        if os.path.exists(h5ad_file_path):
            is_h5ad = True
            logger.info("✅ Loading the data from h5ad file...")
            mtx_data = sc.read_h5ad(h5ad_file_path)
            quant_json_data, permit_list_json_data = json.loads(mtx_data.uns['quant_info']), json.loads(mtx_data.uns['gpl_info'])
            
            feature_dump_data = pd.DataFrame(mtx_data.obs)
            # Standardize feature dump columns to snake_case format
            standardize_feature_dump_columns(feature_dump_data)
            usa_mode = quant_json_data['usa_mode']
            
        else:
            mtx_data = load_fry(mtx_dir_path, output_format='raw')
            # TODO: load the U+S+A mtx data, compute median gene per cell based on mtx
            # USA_mtx_data = load_fry(mtx_dir_path, output_format='all')
            
            # Load  quant.json, generate_permit_list.json, and featureDump.txt
            quant_json_data, permit_list_json_data, feature_dump_data = load_json_txt_file(quant_out_dir, used_simpleaf)
            
            # detect usa_mode
            usa_mode = quant_json_data['usa_mode']
    
    return quant_json_data, permit_list_json_data, feature_dump_data, mtx_data, usa_mode, is_h5ad

def load_json_txt_file(parent_dir):
    """
    Loads quant.json and generate_permit_list.json from the given directory.
    """

    quant_json_data_path = Path(os.path.join(parent_dir, "quant.json"))
    permit_list_path = Path(os.path.join(parent_dir, "generate_permit_list.json"))
    feature_dump_path = Path(os.path.join(parent_dir, "featureDump.txt"))
    
    # Check if quant.json exists
    if not quant_json_data_path.exists():
        logger.error(f"❌ Error: Missing required file: '{quant_json_data_path}'")
        quant_json_data = {}  
    else:
        with open(quant_json_data_path, 'r') as f:
            quant_json_data = json.load(f)

    # Check if generate_permit_list.json exists
    if not permit_list_path.exists():
        # logger.info(f"Permit list file is unavailable.")
        permit_list_json_data = {} 
    else:
        with open(permit_list_path, 'r') as f:
            permit_list_json_data = json.load(f)

    # Check if feature_dump.txt exists
    if not feature_dump_path.exists():
        logger.error(f"❌ Error: Missing required file: '{feature_dump_path}'")
        raise ValueError(f"Missing required file: '{feature_dump_path}'")
    else:
        feature_dump_data = pd.read_csv(feature_dump_path, sep='\t')
        # rename the columns, align with the standard snake_case format
        feature_dump_data.columns = STANDARD_COLUMNS
        logger.debug(f"feature_dump_data columns: {feature_dump_data.columns}")

    return quant_json_data, permit_list_json_data, feature_dump_data

# from https://ga4gh.github.io/refget/seqcols/
def canonical_str(item: [list, dict]) -> bytes:
    """Convert a list or dict into a canonicalized UTF8-encoded bytestring representation"""
    return json.dumps(
        item, separators=(",", ":"), ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode("utf8")

def sha512t24u_digest(seq: bytes) -> str:
    """ GA4GH digest function """
    offset = 24
    digest = hashlib.sha512(seq).digest()
    tdigest_b64us = base64.urlsafe_b64encode(digest[:offset])
    return tdigest_b64us.decode("ascii")

def get_name_digest(item: [list]) -> str:
    return sha512t24u_digest(canonical_str(item))

def get_name_mapping_file_from_registry(seqcol_digest: str, output_dir: Path) -> Path | None:
    """
    Based on `seqcol_digest`, this function will attempt to access the remote registry and 
    look for a known id-to-symbol mapping that matches the digest.  If this is successful, it 
    will download the file and return the path to the downloaded file.  Otherwise, it will return None.
    """
    output_file = output_dir / f"{seqcol_digest}.csv"
    REGISTRY_URL = "https://raw.githubusercontent.com/COMBINE-lab/QCatch-resources/refs/heads/main/resources/registries/id2name.json"
    r = requests.get(REGISTRY_URL)
    if r.ok:
        reg = r.json()
        if seqcol_digest in reg:
            file_url = reg[seqcol_digest]['url']
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

def add_gene_symbol(adata, gene_id2name_file: Path | None, output_dir: Path):
    if adata.var.index.names == ['gene_ids']:
        # from mtx data
        all_gene_ids = adata.var.index
    else:
        # from h5ad data
        if 'gene_id' in adata.var:
            all_gene_ids = adata.var['gene_id']
        elif 'gene_ids' in adata.var:
            # from original simpleaf mtx data
            all_gene_ids = adata.var['gene_ids']
        else:
            logger.error("❌ Error: Neither 'gene_id' nor 'gene_ids' found in adata.var columns; cannot add mapping")
            return
 
    # check the digest for this adata object
    all_gene_ids = pd.Series(all_gene_ids)  # Convert to Series
    seqcol_digest = get_name_digest(list(sorted(all_gene_ids.to_list())))
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
            logger.warning(f"Failed to properly obtain gene id-to-name mapping; will not add mapping")
            return adata
    elif gene_id2name_file.exists() and gene_id2name_file.is_file():
        gene_id2name_path = gene_id2name_file
    else:
        logger.warning(f"If gene id-to-name mapping is provided, it should be a file, but a directory was provided; will not add mapping")
        return adata

    # add the gene symbol, based on the gene id to symbol mapping
    gene_id_to_symbol = pd.read_csv(gene_id2name_path)

    # Identify missing gene symbols
    missing_symbols_count = gene_id_to_symbol['gene_name'].isna().sum()

    if missing_symbols_count > 0:
        logger.info(f"Number of gene IDs with missing gene_name/symbols: {missing_symbols_count}")
        # Replace NaN values in 'gene_symbol' with the corresponding 'gene_id'
        gene_id_to_symbol['gene_name'].fillna(gene_id_to_symbol['gene_id'], inplace=True)
        logger.info(f"fill missing symbols with gene_id.")

    # Create a mapping dictionary from 'query' to 'symbol'
    id_to_symbol_dict = pd.Series(gene_id_to_symbol["gene_name"].values, index=gene_id_to_symbol["gene_id"]).to_dict()

    # Initialize an empty list to hold the reordered symbols
    reordered_symbols = []

    # Iterate through 'all_gene_ids' and fetch corresponding symbols
    for gene_id in all_gene_ids:
        symbol = id_to_symbol_dict.get(gene_id,) 
        reordered_symbols.append(symbol)

    #  Integrate the Reordered Mapping into AnnData
    # Assign gene symbols to AnnData's .var attribute
    adata.var['gene_symbol'] = reordered_symbols

    # (Optional) Replace var_names with gene symbols
    # This can make plots and analyses more interpretable
    adata.var_names = adata.var['gene_symbol'].astype(str)
    # Ensure uniqueness of var_names after replacement
    adata.var_names_make_unique(join="-")
    
    return adata

