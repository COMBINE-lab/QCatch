import argparse
import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import scanpy as sc

from qcatch.fry_tools.loader import load_fry
from qcatch.input_processing import (
    add_gene_symbol,
    load_json_txt_file,
    standardize_feature_dump_columns,
)

logger = logging.getLogger(__name__)


def load_hdf5(
    hdf5_path: Path,
) -> tuple[sc.AnnData, dict, dict, dict | None, pd.DataFrame, bool, str | None]:
    """
    Load an h5ad file and extract quantification, mapping, and feature dump information.

    Parameters
    ----------
    hdf5_path
        Path to the h5ad file.

    Returns
    -------
    tuple
        A tuple containing:
        - AnnData object loaded from the h5ad file.
        - Quantification info.
        - Permit list info.
        - Mapping info if available, otherwise None.
        - Feature dump data.
        - whether USA mode was used.
    """
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


def parse_saved_chem(map_info: str) -> str | None:
    """Parse and standardize chemistry name from simpleaf_map_info output."""
    logger.info("💡 Inferring chemistry from simpleaf_map_info...")
    try:
        cmdline = map_info.get("cmdline")
        mapper = map_info.get("mapper")
        if not cmdline or not mapper:
            raise ValueError("Invalid: map_info must contain 'cmdline' and 'mapper' keys.")

        # Tokenize safely
        try:
            tokens = shlex.split(cmdline)
        except ValueError:
            tokens = cmdline.split()

        # Define allowed and conversion maps
        PISCEM_ALLOWED = {"chromium_v2", "chromium_v3", "chromium_v3_5p", "chromium_v4_3p"}
        SALMON_ALLOWED_FLAGS = {"--chromium"}

        # TODO: 10xv3 and 10xv4-3p both use "--chromiumV3" flag in salmon. but they require different n-partitions in cell calling.
        # SALMON_ALLOWED_FLAGS = {"--chromium", "--chromiumV3"}

        CHEM_CONVERT_MAP = {
            "--chromium": "10X_3p_v2",
            # "--chromiumV3": "10X_3p_v3",
            "chromium_v2": "10X_3p_v2",
            "chromium_v3": "10X_3p_v3",
            "chromium_v3_5p": "10X_5p_v3",
            "chromium_v4_3p": "10X_3p_v4",
        }

        mapper_lower = mapper.lower()
        saved_chem = None

        # piscem parsing
        if mapper_lower == "piscem":
            for i, tok in enumerate(tokens):
                if tok in ("-g", "--geometry") and i + 1 < len(tokens):
                    candidate = tokens[i + 1]
                    if candidate in PISCEM_ALLOWED:
                        saved_chem = candidate
                        break

        # salmon parsing
        elif mapper_lower == "salmon":
            present = [flag for flag in SALMON_ALLOWED_FLAGS if flag in tokens]
            if len(present) == 1:
                saved_chem = present[0]
            elif len(present) > 1:
                raise ValueError(f"Multiple chemistry flags found: {present}")

        else:
            raise ValueError(f"Unknown mapper '{mapper}'.")

        # Convert to standardized label if found
        standardized = CHEM_CONVERT_MAP.get(saved_chem)
        logger.info(
            f"✅ Successfully identified valid chemistry type: {standardized}. "
            "Using this chemistry for subsequent cell calling steps."
        )
        return standardized

    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to identify valid chemistry from simpleaf_map_info: {e}")
        return None


@dataclass
class QuantInput:
    """
    Handle quantification input from a single h5ad file or a directory.

    This class detects whether the input is an h5ad file or a directory
    and loads associated quantification results, mapping logs, and feature data.
    """

    def add_geneid_2_name_if_absent(self, gene_id_2_name_file: Path, output_dir: Path) -> bool:
        """Checks if the underlying dataframe object already has a gene_symbol column and if not, tries to populate it from the gene_id_2_name_dir provided."""
        if "gene_symbol" in self.mtx_data.var.columns:
            self.has_gene_name_mapping = True
            return True
        else:
            self.mtx_data = add_gene_symbol(self.mtx_data, gene_id_2_name_file, output_dir)
            ret = "gene_symbol" in self.mtx_data.var.columns
            self.has_gene_name_mapping = ret
            return ret

    def __init__(self, input_str: str):
        """
        Detect the input format of the quantification output and load associated data.

        Parameters
        ----------
        input_str
            Path to a quantification output directory or an h5ad file.

        Returns
        -------
        Loaded data
        """
        self.provided = Path(input_str)
        if not self.provided.exists():
            raise ValueError(f"The provided input path {self.provided} did not exist")
        # it quant output exists
        if self.provided.is_file():
            self.file = self.provided
            self.dir = self.file.parent
            self.from_simpleaf = True
            self.is_h5ad = True
            logger.info(f"Input {self.provided} inferred to be a file; parent path is {self.dir}")
            logger.info("✅ Loading the data from h5ad file...")
            (
                self.mtx_data,
                self.quant_json_data,
                self.permit_list_json_data,
                self.map_json_data,
                self.feature_dump_data,
                self.usa_mode,
            ) = load_hdf5(self.file)

        else:
            self.dir = self.provided
            logger.info(f"Input {self.provided} inferred to be a directory; searching for valid input file")
            self.mtx_dir_path = None
            self.af_map_path = None
            if os.path.exists(os.path.join(self.dir, "af_quant")) or os.path.exists(
                os.path.join(self.dir, "simpleaf_quant_log.json")
            ):
                logger.info("✅ Detected: 'simpleaf' was used for the quantification result.")
                self.from_simpleaf = True
                self.mtx_dir_path = os.path.join(self.dir, "af_quant")

            elif os.path.exists(os.path.join(self.dir, "alevin")):
                logger.info("✅ Detected: 'alevin-fry' was used for the quantification result.")
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
                logger.error("❌ Error: Expected matrix directory not found in 'af_quant'.")
                self.mtx_data = None

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
            else:
                logger.info("Not finding quants.h5ad file, loading from mtx directory...")
                try:
                    custome_format = {"X": ["S", "A", "U"], "unspliced": ["U"], "spliced": ["S"], "ambiguous": ["A"]}
                    self.mtx_data = load_fry(str(self.mtx_dir_path), output_format=custome_format)
                except Exception:
                    logger.exception("Error calling load_fry")

                self.mtx_data.var["gene_id"] = self.mtx_data.var.index

                # Load quant.json, generate_permit_list.json, and featureDump.txt

                (
                    self.quant_json_data,
                    self.permit_list_json_data,
                    self.feature_dump_data,
                ) = load_json_txt_file(self.mtx_dir_path)

                self.map_json_data = None
                logger.warning(
                    "⚠️ Unfortunately, the mapping log file is not included in output folder if using 'alevin-fry'. As a result, the mapping rate will not be shown in the summary table. However, you can still find this information in your original mapping results from piscem or salmon"
                )

                # detect usa_mode
                self.usa_mode = self.quant_json_data["usa_mode"]
                self.known_chemistry = None


def get_input(input_str: str) -> QuantInput:
    """Wrapper function to instantiate QuantInput from a string input."""
    try:
        return QuantInput(input_str)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid get_input value: {input_str}\n→ {e}") from e
