import os
import pandas as pd
import json
from pyroe import load_fry
import logging
import scanpy as sc

logger = logging.getLogger(__name__)

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
        # rename the columns, align with the featureDump.txt

        feature_dump_data.columns = ['CB', 'CorrectedReads', 'MappedReads', 'DeduplicatedReads', 'MappingRate', 'DedupRate', 'MeanByMax', 'NumGenesExpressed', 'NumGenesOverMean']
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
            # rename the columns, align with the featureDump.txt
            
            feature_dump_data.columns = ['CB', 'CorrectedReads', 'MappedReads', 'DeduplicatedReads', 'MappingRate', 'DedupRate', 'MeanByMax', 'NumGenesExpressed', 'NumGenesOverMean']
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

def load_json_txt_file(quant_out_dir, used_simpleaf):
    """
    Loads quant.json and generate_permit_list.json from the given directory.
    """
    parent_dir = Path(os.path.join(quant_out_dir, "simpleaf_quant", "af_quant")) if used_simpleaf else quant_out_dir

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

    return quant_json_data, permit_list_json_data, feature_dump_data

def add_gene_symbol(adata, gene_id2name_dir):
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
            logger.error("❌ Error: Neither 'gene_id' nor 'gene_ids' found in adata.var columns")
    
    # check the species, then determine the gene_id2name_path
    all_gene_ids = pd.Series(all_gene_ids)  # Convert to Series
    check_first_gene_id = all_gene_ids.iloc[0]  # Now this works
    # check_first_gene_id = all_gene_ids[0]
    if check_first_gene_id.startswith('ENSG'):
        species = 'human'
    elif check_first_gene_id.startswith('ENSMUSG'):
        species = 'mouse'
    else :
        print(f'first gene id: {check_first_gene_id}')
        logger.error("❌ Error: The gene id format is not recognized. We only have human and mouse gene id2name mapping by default.")
        species = 'unknown'
        
    if species == 'unknown':
        return adata
    
    gene_id2name_path = os.path.join(gene_id2name_dir, f'{species}_gene_id2name.csv')
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
    adata.var['symbol'] = reordered_symbols

    # (Optional) Replace var_names with gene symbols
    # This can make plots and analyses more interpretable
    adata.var_names = adata.var['symbol'].astype(str)
    # Ensure uniqueness of var_names after replacement
    adata.var_names_make_unique(join="-")
    
    return adata

