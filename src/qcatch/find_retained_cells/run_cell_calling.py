#!/usr/bin/env python
"""Master function that run calling cell process"""

import logging
import pickle

import numpy as np

from qcatch.logger import QCatchLogger
from qcatch.utils import parse_saved_chem

from .cell_calling import NonAmbientBarcodeResult, find_nonambient_barcodes, initial_filtering_OrdMag
from .matrix import CountMatrix

logger = logging.getLogger("qcatch")
assert isinstance(logger, QCatchLogger), "Logger is not a QCatchLogger. Call setup_logger() in main.py first."


def internal_cell_calling(args, save_for_quick_test, quick_test_mode):
    """
    Perform internal cell calling via initial filtering and non-ambient barcode detection.

    This function runs two main steps:
    1. Initial filtering using OrdMag to identify high-confidence barcodes.
    2. Optional identification of additional non-ambient barcodes using the EmptyDrops algorithm.

    Returns
    -------
    valid_bcs : set
        Set of retained barcode strings.
    intermediate_result : tuple
        Tuple of (converted_filtered_bcs, non_ambient_result) used in downstream output generation.
    """
    matrix = CountMatrix.from_anndata(args.input.mtx_data)
    chemistry = args.chemistry
    n_partitions = args.n_partitions
    verbose = args.verbose
    if chemistry is None and n_partitions is None:
        # infer chemistry from metadata
        map_json_data = args.input.map_json_data
        known_chemistry = parse_saved_chem(map_json_data) if map_json_data else None
        if known_chemistry is None:
            msg = (
                "❌ Required parameter missing: at least one of 'chemistry' or 'n_partitions' must be provided.\n"
                "Please specify either the chemistry version (via --chemistry / -c) "
                "or the number of partitions (via --n_partitions / -n)."
            )
            logger.error(msg)
            raise SystemExit(1)
        else:
            chemistry = known_chemistry
    # # cell calling step1 - empty drop
    logger.info("🧬 Starting cell calling...")
    filtered_bcs = initial_filtering_OrdMag(matrix, chemistry, n_partitions, verbose)
    logger.info(f"🧀 Step1- number of inital filtered cells: {len(filtered_bcs)}")
    converted_filtered_bcs = [x.decode() if isinstance(x, np.bytes_ | bytes) else str(x) for x in filtered_bcs]
    non_ambient_result = None
    valid_bcs = set(converted_filtered_bcs)
    output_dir = args.output
    if quick_test_mode:
        # Re-load the saved result from pkl file
        with open(f"{output_dir}/non_ambient_result.pkl", "rb") as f:
            non_ambient_result = pickle.load(f)
    else:
        # cell calling step2 - empty drop
        non_ambient_result: NonAmbientBarcodeResult | None = find_nonambient_barcodes(
            matrix, filtered_bcs, chemistry, n_partitions, verbose=verbose
        )

    if non_ambient_result is None:
        non_ambient_cells = 0
        logger.record_warning(
            "⚠️ Warning❗️: Step2- Empty drop failed: non_ambient_result is None. This may indicate low data quality, an incomplete input matrix, or an incorrect chemistry version."
        )

    else:
        non_ambient_cells = len(non_ambient_result.eval_bcs)
        logger.debug(f"🍧 Step2- Empty drop: number of all potential non-ambient cells: {non_ambient_cells}")
        if save_for_quick_test:
            with open(f"{output_dir}/non_ambient_result.pkl", "wb") as f:
                pickle.dump(non_ambient_result, f)

        # extract the non-ambient cells from eval_bcs from a binary array
        is_nonambient_bcs = [
            str(bc)
            for bc, boolean_non_ambient in zip(
                non_ambient_result.eval_bcs, non_ambient_result.is_nonambient, strict=False
            )
            if boolean_non_ambient
        ]
        logger.info(f"🍹 Step2- empty drop: number of is_non_ambient cells: {len(is_nonambient_bcs)}")

        # Calculate the total number of valid barcodes
        valid_bcs = set(converted_filtered_bcs) | set(is_nonambient_bcs)
        # num of all processed cells
        all_cells = args.input.mtx_data.shape[0]
        # Save the total retained cells to a txt file
        logger.info(f"✅ Retained cells after two-step cell calling: {len(valid_bcs)} out of {all_cells} cells")

    intermediate_result = (converted_filtered_bcs, non_ambient_result)
    return valid_bcs, intermediate_result
