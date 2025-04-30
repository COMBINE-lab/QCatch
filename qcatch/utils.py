from pathlib import Path
import scanpy as sc
import pandas as pd
import json
import os
from dataclasses import dataclass
from pyroe import load_fry
from typing import List
import argparse
from qcatch.input_processing import load_json_txt_file, add_gene_symbol, STANDARD_COLUMNS, CAMEL_TO_SNAKE_MAPPING, standardize_feature_dump_columns


import logging

logger = logging.getLogger(__name__)

def load_hdf5(hdf5_path: Path) -> sc.AnnData:
    mtx_data = sc.read_h5ad(hdf5_path)
    quant_json_data, permit_list_json_data = (
        json.loads(mtx_data.uns["quant_info"]),
        json.loads(mtx_data.uns["gpl_info"]),
    )
    map_json_data = json.loads(mtx_data.uns["simpleaf_map_info"]) if "simpleaf_map_info" in mtx_data.uns else None

    feature_dump_data = pd.DataFrame(mtx_data.obs)
    feature_dump_data = standardize_feature_dump_columns(feature_dump_data)
    usa_mode = quant_json_data["usa_mode"]

    return mtx_data, quant_json_data, permit_list_json_data, map_json_data, feature_dump_data, usa_mode


@dataclass
class QuantInput:
    def add_geneid_2_name_if_absent(
        self, gene_id_2_name_file: Path, output_dir: Path
    ) -> bool:
        """
        Checks if the underlying dataframe object already has a gene_symbol column and
        if not, tries to populate it from the gene_id_2_name_dir provided
        """
        if "gene_symbol" in self.mtx_data.var.columns:
            self.has_gene_name_mapping = True
            return True
        else:
            self.mtx_data = add_gene_symbol(
                self.mtx_data, gene_id_2_name_file, output_dir
            )
            ret = "gene_symbol" in self.mtx_data.var.columns
            self.has_gene_name_mapping = ret
            return ret

    def __init__(self, input_str: str):
        """
        Detects the input format of the quantification output directory.
        return the loaded data
        """
        self.provided = Path(input_str)
        if not self.provided.exists():
            raise ValueError(f"The provided input path {self.provided} did not exist")
        # it exists
        if self.provided.is_file():
            self.file = self.provided
            self.dir = self.file.parent
            self.from_simpleaf = True
            self.is_h5ad = True
            logger.info(
                f"Input {self.provided} inferred to be a file; parent path is {self.dir}"
            )
            logger.info("✅ Loading the data from h5ad file...")
            (
                self.mtx_data,
                self.quant_json_data,
                self.permit_list_json_data,
                self.map_json_data,
                self.feature_dump_data,
                self.usa_mode,
            ) = load_hdf5(self.file)

            # TODO: deprecated later, when h5ad has the mapping info
            if self.map_json_data is None:
                self.map_json_data = find_mapping_info(self.dir.parent)

        else:
            self.dir = self.provided
            logger.info(
                f"Input {self.provided} inferred to be a directory; searching for valid input file"
            )
            self.mtx_dir_path = None
            self.af_map_path = None
            if os.path.exists(os.path.join(self.dir, "af_quant")) or os.path.exists(os.path.join(self.dir, "simpleaf_quant_log.json")):
                logger.info(
                    "✅ Detected: 'simpleaf' was used for the quantification result."
                )
                self.from_simpleaf = True
                self.mtx_dir_path = os.path.join(self.dir, "af_quant")

            elif os.path.exists(os.path.join(self.dir, "alevin")):
                logger.info(
                    "✅ Detected: 'alevin-fry' was used for the quantification result."
                )
                self.from_simpleaf = False
                self.mtx_dir_path = self.dir
            else:
                logger.warning(
                    "⚠️ Unable to recognize the quantification directory. "
                    "Ensure that the directory structure remains unchanged from the original output directory."
                )
                

            # -----------------------------------
            # Loads matrix data from the given quantification output directory.

            if not self.mtx_dir_path:
                logger.error(
                    "❌ Error: Expected matrix directory not found in 'af_quant'."
                )
                mtx_data = None

            self.is_h5ad = False
            # -----------------------------------
            # Check if quants.h5ad file exists in the parent directory
            h5ad_file_path = os.path.join(self.mtx_dir_path, "alevin", "quants.h5ad")
            if os.path.exists(h5ad_file_path):
                self.file = h5ad_file_path
                self.is_h5ad = True
                logger.info("✅ Loading the data from h5ad file...")
                (
                    self.mtx_data,
                    self.quant_json_data,
                    self.permit_list_json_data,
                    self.map_json_data,
                    self.feature_dump_data,
                    self.usa_mode,
                ) = load_hdf5(self.file)
                # TODO: deprecated later, when h5ad has the mapping info
                if self.map_json_data is None:
                    self.map_json_data = find_mapping_info(self.dir)
 
            else:
                logger.info("Not finding quants.h5ad file, loading from mtx directory...")
                try:
                    custome_format ={'X' : ['S', 'A', 'U'], 'unspliced' : ['U'], 'spliced' : ['S'], 'ambiguous' : ['A']}
                    self.mtx_data = load_fry(str(self.mtx_dir_path), output_format=custome_format)
                except Exception as e:
                    logger.error(f"Error calling load_fry :: {e}")
                
                self.mtx_data.var["gene_id"] = self.mtx_data.var.index

                # Load quant.json, generate_permit_list.json, and featureDump.txt

                (
                    self.quant_json_data,
                    self.permit_list_json_data,
                    self.feature_dump_data,
                ) = load_json_txt_file(self.mtx_dir_path)
                
                self.map_json_data = None
                logger.warning("⚠️ Unfortunately, the mapping log file is not included in output folder if using 'alevin-fry'. As a result, the mapping rate will not be shown in the summary table. However, you can still find this information in your original mapping results from piscem or salmon")
        
                # detect usa_mode
                self.usa_mode = self.quant_json_data["usa_mode"]


def get_input(input_str: str) -> QuantInput:
    try:
        return QuantInput(input_str)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid get_input value: {input_str}\n→ {e}")


def find_mapping_info(parent_quant_dir):
    # find the map_json file
    map_json_path_1 = os.path.join(parent_quant_dir, "af_map", "aux_info", "map.json")
    map_json_path_2 = os.path.join(parent_quant_dir, "af_map", "map_info.json")
    if os.path.exists(map_json_path_1):
        map_json_path = map_json_path_1
    elif os.path.exists(map_json_path_2):
        map_json_path = map_json_path_2
    else:
        logger.warning("⚠️  Mapping log file not found. Mapping rate will not be displayed in the summary table.")
        return None
    
    with open(map_json_path, 'r') as f:
        map_json_data = json.load(f)
            
    return map_json_data