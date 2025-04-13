import os
import argparse
from bs4 import BeautifulSoup
import importlib.resources as pkg_resources
from pathlib import Path
import numpy as np
import pickle
import logging
import shutil

from qcatch import templates
from qcatch.utils import QuantInput, get_input
from qcatch.plots_tables import show_quant_log_table
from qcatch.convert_plots import create_plotly_plots, modify_html_with_plots
from qcatch.find_retained_cells.matrix import CountMatrix
from qcatch.find_retained_cells.cell_calling import initial_filtering_OrdMag, find_nonambient_barcodes, NonAmbientBarcodeResult


def load_template():
    # Open the template file and parse it with BeautifulSoup
    template_path = pkg_resources.files(templates) / 'report_template.html'
    with open(template_path, encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    return soup

def main():
    # Remove all existing handlers from the root logger
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    # Set logging level based on the verbose flag
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s :\n %(message)s"
    )

    parser = argparse.ArgumentParser(description="QCatch: Command-line Interface")
    # Add command-line arguments
    parser.add_argument(
        '--input', '-i', 
        type=get_input, 
        required=True, 
        help="Path to the input directory containing the quantification output files or to the HDF5 file itself."
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        required=True,
        help="Path to the desired output directory."
    )

    parser.add_argument(
        '--chemistry', '-c', 
        type=str, 
        help="Specifies the chemistry used in the experiment, which determines the range for the empty_drops step. Options: '10X_3p_v2', '10X_3p_v3', '10X_3p_v4', '10X_5p_v3', '10X_3p_LT', '10X_HT'. If not provided, we'll use the default range (which is the range used for '10X_3p_v2' and '10X_3p_v3')."
    )
    parser.add_argument(
        '--n_partitions', '-n', 
        type=int, 
        default=None,
        help="Number of partitions (max number of barcodes to consider for ambient estimation). Skip this step if you already specify the chemistry. Otherwise, you can specify the desired `n_partitions`. "
    )
    
    parser.add_argument(
        '--gene_id2name_file', '-g', 
        type=Path,
        default=None,
        help="(Optional) File provides a mapping from gene IDs to gene names. The file must be a CSV file contains two columns with headers: 'gene_id' (e.g., ENSG00000284733) and 'gene_name' (e.g., OR4F29). If not provided, an attempt will be made to look up the mapping in a remote registry. If that lookup fails the mitochondria plots will not be displayed."
    )
    parser.add_argument(
        '--save_filtered_h5ad', '-s',
        action='store_true',
        help="If enabled, `qcatch` will save a separate `.h5ad` file containing only the retained cells."
    )
    parser.add_argument(
        '--overwrite_h5ad', '-w',
        action='store_true',
        help="If enabled, `qcatch` will overwrite the original `.h5ad` file in place by appending cell filtering results to anndata.obs. No existing data or cells will be removed; only additional metadata columns are added."
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true', 
        help='Enable verbose logging with debug level messages')

    args = parser.parse_args()

    # Set logging level based on the verbose flag
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s :\n %(message)s"
    )
    
    input_dir = args.input
    output_dir = args.output
    chemistry = args.chemistry 
    n_partitions = args.n_partitions
    gene_id2name_file = args.gene_id2name_file
    verbose = args.verbose
    overwrite_h5ad = args.overwrite_h5ad
    save_filtered_h5ad = args.save_filtered_h5ad
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    
    # Suppress Numba’s debug messages by raising its level to WARNING
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    # Set up logging
    logger = logging.getLogger(__name__)
   
    # # Cell calling, get the number of non-ambient barcodes
    matrix = CountMatrix.from_anndata(args.input.mtx_data)

    # add gene_id_2_name if we don't yet have it
    output_path = Path(args.output)
    args.input.add_geneid_2_name_if_absent(gene_id2name_file, output_path)
    
    # # cell calling step1 - empty drop
    logger.info("🧬 Starting cell calling...")
    filtered_bcs = initial_filtering_OrdMag(matrix, chemistry, n_partitions)
    logger.info(f"🔎 step1- number of inital filtered cells: {len(filtered_bcs)}")
    converted_filtered_bcs =  [x.decode() if isinstance(x, np.bytes_) else str(x) for x in filtered_bcs]
    non_ambient_result =None
    
    quick_test_mode_init = False
    quick_test_mode = False
    if quick_test_mode:
        # Re-load the saved result from pkl file
        with open(f'{output_dir}/non_ambient_result.pkl', 'rb') as f:
            non_ambient_result = pickle.load(f)
    else:
        # cell calling step2 - empty drop
        non_ambient_result : NonAmbientBarcodeResult | None = find_nonambient_barcodes(matrix, filtered_bcs, chemistry, n_partitions, verbose = verbose)
    
    if non_ambient_result is None:
        non_ambient_cells = 0
        valid_bcs = set(converted_filtered_bcs) 
        logger.warning(" ⚠️ non_ambient_result is None. Please verify the chemistry version or ensure that the input matrix is complete.")
        
    else:
        non_ambient_cells = len(non_ambient_result.eval_bcs)
        logger.debug(f"step2- Empty drop: number of all potential non-ambient cells: {non_ambient_cells}")
        if quick_test_mode_init:
            with open(f'{output_dir}/non_ambient_result.pkl', 'wb') as f:
                pickle.dump(non_ambient_result, f)
        
        # extract the non-ambient cells from eval_bcs from a binary array
        is_nonambient_bcs = [str(bc) for bc, boolean_non_ambient in zip(non_ambient_result.eval_bcs, non_ambient_result.is_nonambient) if boolean_non_ambient]
        logger.info(f"🔎 step2- empty drop: number of is_non_ambient cells: {len(is_nonambient_bcs)}")
        
        # Calculate the total number of valid barcodes
        valid_bcs = set(converted_filtered_bcs) | set(is_nonambient_bcs)
        
        # Save the total retained cells to a txt file
        logger.info(f"✅ Total reatined cells after cell calling: {len(valid_bcs)}")
        
        if args.input.is_h5ad:
            # Update the h5ad file with the final retain cells, contains original filtered cells and passed non-ambient cells
            args.input.mtx_data.obs['initial_filtered_cell'] = args.input.mtx_data.obs['barcodes'].isin(converted_filtered_bcs)
            args.input.mtx_data.obs['potential_non_ambient_cell'] = args.input.mtx_data.obs['barcodes'].isin(non_ambient_result.eval_bcs)
            
            # Create a mapping from barcodes to p-values
            barcode_to_pval = dict(zip(non_ambient_result.eval_bcs, non_ambient_result.pvalues))
            # Assign p-values only where 'is_nonambient' is True, otherwise fill with NaN
            args.input.mtx_data.obs['non_ambient_pvalue'] = args.input.mtx_data.obs['barcodes'].map(barcode_to_pval).astype('float')
            
            args.input.mtx_data.obs['is_retained_cells'] = args.input.mtx_data.obs['barcodes'].isin(valid_bcs)
            
            logger.info("🗂️ Saved 'cell calling result' to the h5ad file, check the new added columns in adata.obs .")
            temp_file = os.path.join(args.input.dir, 'quants_after_QC.h5ad')
            # Save the modified file to a temporary file first
            args.input.mtx_data.write_h5ad(temp_file, compression='gzip')
            if overwrite_h5ad:
                # After successful saving, remove or rename the original
                input_h5ad_file = os.path.join(input_dir, 'quants.h5ad')
                # Delete original file
                os.remove(input_h5ad_file) 

                # Rename temporary file to original filename
                shutil.move(temp_file, input_h5ad_file)
                logger.info(f"📋 Overwrited the original h5ad file with the new cell calling result.")
            if save_filtered_h5ad:
                # filter the anndata , only keep the cells in valid_bcs
                filter_mtx_data = args.input.mtx_data[args.input.mtx_data.obs['is_retained_cells'].values, :].copy()
                # Save the filtered anndata to a new file
                filter_mtx_data_filename = os.path.join(output_dir, 'filtered_quants.h5ad')
                filter_mtx_data.write_h5ad(filter_mtx_data_filename, compression='gzip')
                logger.info(f"📋 Saved the filtered h5ad file to {filter_mtx_data_filename}.")
            
        else:
            # Not h5ad file, write to new files
            # 1- original filtered cells
            initial_filtered_cells_filename= os.path.join(output_dir,'initial_filtered_cells.txt' )
            
            with open(initial_filtered_cells_filename, 'w') as f:
                for bc in converted_filtered_bcs:
                    f.write(f"{bc}\n")
            
            # 2- additional non-ambient cells results
            # Save barcode and adjusted p-values to a txt file
            pval_output_file = os.path.join(output_dir, 'potential_nonambient_result.txt')
            with open(pval_output_file, 'w') as f:
                f.write("barcodes\tadj_pval\n")
                for bc, pval in zip(non_ambient_result.eval_bcs, non_ambient_result.pvalues):
                    f.write(f"{bc}\t{pval}\n")
            
            # Save the total retained cells to a txt file
            total_retained_cell_file = f'{output_dir}/total_retained_cells.txt'
            with open(total_retained_cell_file, 'w') as f:
                for bc in valid_bcs:
                    f.write(f"{bc}\n")
                    
            # Logging the cell calling result path
            logger.info(f'🗂️ Saved cell calling result in the output directory: {output_dir}')
    
    # if non_ambient_result is not None:
    #     # Load the result from pkl file
    #     with open(f'{output_dir}/non_ambient_result.pkl', 'rb') as f:
    #         non_ambient_result = pickle.load(f)

    # NOTE: The h5ad file has already been saved (if applicable).
    # Any further modifications to `adata` below (for plotting purposes) 
    # will not affect the h5ad files in disk.
    
    # plots and log, summary tables
    plot_text_elements = create_plotly_plots(args.input.feature_dump_data, args.input.mtx_data, valid_bcs, args.input.usa_mode, args.input.is_h5ad)
    
    quant_json_table_html, permit_list_table_html = show_quant_log_table(args.input.quant_json_data, args.input.permit_list_json_data)


    # Modify HTML with plots
    modify_html_with_plots(
        # report template
        soup=load_template(),
        output_html_path=os.path.join(output_dir, f'QCatch_report.html'),
        plot_text_elements = plot_text_elements,
        quant_json_table_html = quant_json_table_html,
        permit_list_table_html = permit_list_table_html,
        usa_mode=args.input.usa_mode
    )

if __name__ == "__main__":
    main()
