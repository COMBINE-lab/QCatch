from pathlib import Path
import scanpy as sc
import pandas as pd
import json
import os
from dataclasses import dataclass
from pyroe import load_fry

from QCatch.input_processing import load_json_txt_file, add_gene_symbol

import logging

logger = logging.getLogger(__name__)


def load_hdf5(hdf5_path: Path) -> sc.AnnData:
    mtx_data = sc.read_h5ad(hdf5_path)
    quant_json_data, permit_list_json_data = (
        json.loads(mtx_data.uns["quant_info"]),
        json.loads(mtx_data.uns["gpl_info"]),
    )

    feature_dump_data = pd.DataFrame(mtx_data.obs)
    # rename the columns, align with the featureDump.txt
    feature_dump_data.columns = [
        "CB",
        "CorrectedReads",
        "MappedReads",
        "DeduplicatedReads",
        "MappingRate",
        "DedupRate",
        "MeanByMax",
        "NumGenesExpressed",
        "NumGenesOverMean",
    ]
    usa_mode = quant_json_data["usa_mode"]

    return mtx_data, quant_json_data, permit_list_json_data, feature_dump_data, usa_mode


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
                self.feature_dump_data,
                self.usa_mode,
            ) = load_hdf5(self.file)

        else:
            self.dir = self.provided
            logger.info(
                f"Input {self.provided} inferred to be a directory; searching for valid input file"
            )
            if os.path.exists(os.path.join(self.dir, "simpleaf_quant")):
                logger.info(
                    "✅ Detected: 'simpleaf' was used for the quantification result."
                )
                self.from_simpleaf = True
            elif os.path.exists(os.path.join(self.dir, "quant.json")):
                logger.info(
                    "✅ Detected: 'alevin-fry' was used for the quantification result."
                )
                self.from_simpleaf = False
            else:
                logger.warning(
                    "⚠️ Unable to recognize the quantification directory. "
                    "Ensure that the directory structure remains unchanged from the original output directory."
                )

            # -----------------------------------
            # Loads matrix data from the given quantification output directory.
            mtx_dir_path = Path(
                os.path.join(self.dir, "simpleaf_quant", "af_quant")
                if self.from_simpleaf
                else self.dir
            )

            if not mtx_dir_path.exists():
                logger.error(
                    f"❌ Error: Expected matrix directory '{mtx_dir_path}' not found. Please check the input directory structure."
                )
                mtx_data = None

            self.is_h5ad = False
            # -----------------------------------
            # Check if quants.h5ad file exists in the parent directory
            h5ad_file_path = os.path.join(self.dir, "quants.h5ad")
            if os.path.exists(h5ad_file_path):
                self.is_h5ad = True
                logger.info("✅ Loading the data from h5ad file...")
                (
                    self.mtx_data,
                    self.quant_json_data,
                    self.permit_list_json_data,
                    self.feature_dump_data,
                    self.usa_mode,
                ) = load_hdf5(self.file)
            else:
                try:
                    self.mtx_data = load_fry(str(mtx_dir_path), output_format="raw")
                except Exception as e:
                    logger.error(f"Error calling load_fry :: {e}")
                # TODO: load the U+S+A mtx data, compute median gene per cell based on mtx
                # USA_mtx_data = load_fry(mtx_dir_path, output_format='all')
                self.mtx_data.var["gene_id"] = self.mtx_data.var.index

                # Load  quant.json, generate_permit_list.json, and featureDump.txt
                (
                    self.quant_json_data,
                    self.permit_list_json_data,
                    self.feature_dump_data,
                ) = load_json_txt_file(self.dir, self.from_simpleaf)

                # detect usa_mode
                self.usa_mode = self.quant_json_data["usa_mode"]


def get_input(input_str: str) -> QuantInput:
    return QuantInput(input_str)
