#!/usr/bin/env python
"""Master function that run calling cell process"""

import logging
import os
import pickle
import shutil

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
        logger.info(f"✅ Total reatined cells after cell calling: {len(valid_bcs)} out of {all_cells} cells")

    intermediate_result = (converted_filtered_bcs, non_ambient_result)
    return valid_bcs, intermediate_result


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
    save_filtered_h5ad = args.save_filtered_h5ad
    output_dir = args.output
    # Save the cell calling result
    if args.input.is_h5ad:
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
            args.input.mtx_data.obs.drop(
                columns=existing_cols.intersection(args.input.mtx_data.obs.columns), inplace=True
            )

        if args.valid_cell_list:
            # if the user provided a valid cell list
            args.input.mtx_data.obs["is_retained_cells"] = args.input.mtx_data.obs["barcodes"].isin(set(valid_bcs))

            logger.info(
                "🗂️ Saved the ‘cell calling result’ based on the user-specified barcode list to the modified .h5ad file. Check the newly added column in adata.obs. Note: Only one column, 'is_retained_cells', is added. FDR-related information from the internal cell calling process is excluded"
            )

        else:
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

            args.input.mtx_data.obs["is_retained_cells"] = args.input.mtx_data.obs["barcodes"].isin(valid_bcs)

            logger.info(
                "🗂️ Saved 'cell calling result' to the modified h5ad file, check the new added columns in adata.obs ."
            )

        args.input.mtx_data.uns["qc_info"] = qcatch_log

        if output_dir == args.input.dir:
            # Inplace overwrite: same location as original
            temp_file = os.path.join(output_dir, "quants_after_QC.h5ad")
            args.input.mtx_data.write_h5ad(temp_file, compression="gzip")
            input_h5ad_file = args.input.file
            os.remove(input_h5ad_file)
            shutil.move(temp_file, input_h5ad_file)
            logger.info("📋 Overwrote the original h5ad file with the new cell calling result.")
        else:
            # Save to separate file in specified output dir
            output_h5ad_file = os.path.join(output_dir, "quants_after_QC.h5ad")
            args.input.mtx_data.write_h5ad(output_h5ad_file, compression="gzip")
            logger.info(f"📋 Saved modified h5ad file to: {output_h5ad_file}")

        if save_filtered_h5ad:
            # filter the anndata , only keep the cells in valid_bcs
            filter_mtx_data = args.input.mtx_data[args.input.mtx_data.obs["is_retained_cells"].values, :].copy()
            # Save the filtered anndata to a new file
            filter_mtx_data_filename = os.path.join(output_dir, "filtered_quants.h5ad")
            filter_mtx_data.write_h5ad(filter_mtx_data_filename, compression="gzip")
            logger.info(f"📋 Saved the filtered h5ad file to {filter_mtx_data_filename}.")

    else:
        # Not h5ad file, write to new files
        if args.valid_cell_list:
            # if the user provided a valid cell list
            logger.info(
                "🗂️ Skipped saving the cell calling results because the specified cell barcode list already exists."
            )
        else:
            # 1- original filtered cells
            initial_filtered_cells_filename = os.path.join(output_dir, "initial_filtered_cells.txt")
            type_list = []
            for bc in converted_filtered_bcs:
                type_list.append(type(bc))

            print(f"set type of initial c b list {set(type_list)}")
            with open(initial_filtered_cells_filename, "w") as f:
                for bc in converted_filtered_bcs:
                    f.write(f"{bc}\n")

            # 2- additional non-ambient cells results
            if non_ambient_result is not None:
                # Save barcode and adjusted p-values to a txt file
                pval_output_file = os.path.join(output_dir, "potential_nonambient_result.txt")
                with open(pval_output_file, "w") as f:
                    f.write("barcodes\tadj_pval\n")
                    for bc, pval in zip(non_ambient_result.eval_bcs, non_ambient_result.pvalues, strict=False):
                        f.write(f"{bc}\t{pval}\n")

            # Save the total retained cells to a txt file
            total_retained_cell_file = os.path.join(output_dir, "total_retained_cells.txt")
            with open(total_retained_cell_file, "w") as f:
                for bc in valid_bcs:
                    f.write(f"{bc}\n")
            # Logging the cell calling result path
            logger.info(f"🗂️ Saved cell calling result and qcatch log file in the output directory: {output_dir}")
        # Save the qcatch log file. abou the version
        qcatch_log_file = os.path.join(output_dir, "qcatch_log.txt")
        with open(qcatch_log_file, "w") as f:
            for key, value in qcatch_log.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"🗂️ Saved qcatch log file in the output directory: {output_dir}")


def run_cell_calling(args, version, save_for_quick_test, quick_test_mode):
    """Run the cell calling process."""
    if args.valid_cell_list:
        # If a valid cell list is provided, we will skip the cell calling step
        logger.debug("🍻 Using user-specified valid cell list.")
        # parse the valid cell list
        with open(args.valid_cell_list) as f:
            valid_bcs = list({line.strip() for line in f})
        intermediate_result = None
        logger.info(f"🧃 Number of cells found in provided valid cell list : {len(valid_bcs)}")

    else:
        valid_bcs, intermediate_result = internal_cell_calling(args, save_for_quick_test, quick_test_mode)

    # save results
    save_results(args, version, intermediate_result, valid_bcs)

    return valid_bcs
