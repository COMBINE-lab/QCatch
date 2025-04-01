import os
import argparse
from bs4 import BeautifulSoup
import importlib.resources as pkg_resources
from pathlib import Path
import numpy as np
import pickle
import logging
import shutil

from QCatch import templates
from QCatch.plots_tables import show_quant_log_table
from QCatch.input_processing import parse_quant_out_dir
from QCatch.convert_plots import create_plotly_plots, modify_html_with_plots
from QCatch.find_retained_cells.matrix import CountMatrix
from QCatch.find_retained_cells.cell_calling import initial_filtering_OrdMag, find_nonambient_barcodes, NonAmbientBarcodeResult


def load_template():
    # Open the template file and parse it with BeautifulSoup
    template_path = pkg_resources.files(templates) / 'report_template.html'
    with open(template_path, encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    return soup

def main():
    
    parser = argparse.ArgumentParser(description="QCatch: Command-line Interface")
    # Add command-line arguments
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True, 
        help="Path to the input directory containing the quant output files"
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        help="Path to the output directory (default: same directory as input file)"
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
        '--gene_id2name_dir', '-g', 
        type=str,
        default=None,
        help="(Optional) Directory containing the 'gene_id2name' file for converting Ensembl gene IDs to gene names. The file must be a CSV with two columns: 'gene_id' (e.g., ENSG00000284733) and 'gene_name' (e.g., OR4F29). If not provided, mitochondria plots will not be displayed."
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true', 
        help='Enable verbose logging with debug level messages')
    
    parser.add_argument(
        '--overwrite_h5ad', '-w',
        action='store_true',
        help="If set, modifies the original .h5ad file in place by overwriting it with the updated cell filtering results."
    )

    args = parser.parse_args()
    # Set default output directory to the input file's directory
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.as_posix()
    
    input_dir = args.input
    output_dir = args.output
    chemistry = args.chemistry 
    n_partitions = args.n_partitions
    gene_id2name_dir = args.gene_id2name_dir
    verbose = args.verbose
    overwrite_h5ad = args.overwrite_h5ad
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    
    # Remove all existing handlers from the root logger
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    # Set logging level based on the verbose flag
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s :\n %(message)s"
    )
    # Suppress Numba’s debug messages by raising its level to WARNING
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Parse and prepare the input data
    quant_json_data, perimit_list_json_data, feature_dump_data, mtx_data, usa_mode, is_h5ad = parse_quant_out_dir(input_dir)
    
    # # Cell calling, get the number of non-ambient barcodes
    matrix = CountMatrix.from_anndata(mtx_data)
    
    # # cell calling step1 - empty drop
    logger.info("🧬 Starting cell calling...")
    filtered_bcs = initial_filtering_OrdMag(matrix, chemistry, n_partitions)
    logger.info(f"🔎 step1- number of inital filtered cells: {len(filtered_bcs)}")
    converted_filtered_bcs =  [x.decode() if isinstance(x, np.bytes_) else str(x) for x in filtered_bcs]
    
    # # cell calling step2 - empty drop
    non_ambient_result : NonAmbientBarcodeResult | None = find_nonambient_barcodes(matrix, filtered_bcs, chemistry, n_partitions, verbose = verbose)
    
    # # Re-load the saved result from pkl file
    # with open(f'{output_dir}/non_ambient_result.pkl', 'rb') as f:
    #     non_ambient_result = pickle.load(f)
    
    if non_ambient_result is None:
        non_ambient_cells = 0
        valid_bcs = set(converted_filtered_bcs) 
        logger.warning(" ⚠️ non_ambient_result is None. Please verify the chemistry version or ensure that the input matrix is complete.")
        
    else:
        non_ambient_cells = len(non_ambient_result.eval_bcs)
        logger.debug(f"step2- Empty drop: number of all potential non-ambient cells: {non_ambient_cells}")
        with open(f'{output_dir}/non_ambient_result.pkl', 'wb') as f:
            pickle.dump(non_ambient_result, f)
        
        # extract the non-ambient cells from eval_bcs from a binary array
        is_nonambient_bcs = [str(bc) for bc, boolean_non_ambient in zip(non_ambient_result.eval_bcs, non_ambient_result.is_nonambient) if boolean_non_ambient]
        logger.info(f"🔎 step2- empty drop: number of is_non_ambient cells: {len(is_nonambient_bcs)}")
        
        # Calculate the total number of valid barcodes
        valid_bcs = set(converted_filtered_bcs) | set(is_nonambient_bcs)
        
        # Save the total retained cells to a txt file
        logger.info(f"✅ Total reatined cells after cell calling: {len(valid_bcs)}")
        total_retained_cell_file = f'{output_dir}/total_retained_cells.txt'
        with open(total_retained_cell_file, 'w') as f:
            for bc in valid_bcs:
                f.write(f"{bc}\n")
        
        if is_h5ad:
            # Update the h5ad file with the number of expected cells, contains original filtered cells and non-ambient cells
            mtx_data.obs['initial_filtered_cell'] = mtx_data.obs['barcodes'].isin(converted_filtered_bcs)
            mtx_data.obs['additional_non_ambient_cell'] = mtx_data.obs['barcodes'].isin(non_ambient_result.eval_bcs)
            # is non_ambient is True, fill with pvalue, otherwise fill with NaN
            # Create a mapping from barcodes to p-values
            barcode_to_pval = dict(zip(non_ambient_result.eval_bcs, non_ambient_result.pvalues))

            # Assign p-values only where 'non_ambient' is True, otherwise fill with NaN
            mtx_data.obs['non_ambient_pvalue'] = mtx_data.obs['barcodes'].map(barcode_to_pval)
            
            mtx_data.obs['is_retained_cells'] = mtx_data.obs['barcodes'].isin(valid_bcs)
            logger.info("🗂️ Saved 'cell calling result' to the h5ad file, check the new added columns in adata.obs .")
            temp_file = os.path.join(input_dir, 'quants_with_cell_calling_info.h5ad')
            # Save the modified file to a temporary file first
            mtx_data.write_h5ad(temp_file, compression='gzip')
            if overwrite_h5ad:
                # After successful saving, remove or rename the original
                input_h5ad_file = os.path.join(input_dir, 'quants.h5ad')
                # Delete original file
                os.remove(input_h5ad_file) 

                # Rename temporary file to original filename
                shutil.move(temp_file, input_h5ad_file)
                logger.info(f"📋 Overwrited the original h5ad file with the new cell calling result.")
            
        else:
            # Not h5ad file, write to new files
            # 1- original filtered cells
            initial_filtered_cells_filename= os.path.join(output_dir,'initial_filtered_cells.txt' )
            
            with open(initial_filtered_cells_filename, 'w') as f:
                for bc in converted_filtered_bcs:
                    f.write(f"{bc}\n")
            
            # 2- additional non-ambient cells and pvalues
            non_ambient_result_filename=os.path.join(output_dir, 'potential_nonambient_result.pkl')
            
            with open(non_ambient_result_filename, 'wb') as f:
                pickle.dump(non_ambient_result, f)
            # save the cell calling result
            logger.info(f'🗂️ Saved cell calling result in the output directory: {output_dir}')
    
    # if non_ambient_result is not None:
    #     # Load the result from pkl file
    #     with open(f'{output_dir}/non_ambient_result.pkl', 'rb') as f:
    #         non_ambient_result = pickle.load(f)

    # plots and log, summary tables
    plot_text_elements = create_plotly_plots(feature_dump_data, mtx_data, valid_bcs, gene_id2name_dir, usa_mode)
    
    quant_json_table_html, permit_list_table_html = show_quant_log_table(quant_json_data, perimit_list_json_data)


    # Modify HTML with plots
    modify_html_with_plots(
        # report template
        soup=load_template(),
        output_html_path=os.path.join(output_dir, f'{output_dir}/QCatch_report.html'),
        plot_text_elements = plot_text_elements,
        quant_json_table_html = quant_json_table_html,
        permit_list_table_html = permit_list_table_html,
        usa_mode=usa_mode
    )

if __name__ == "__main__":
    main()
