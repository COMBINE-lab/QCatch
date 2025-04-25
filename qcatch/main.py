import os
import argparse
from bs4 import BeautifulSoup
import importlib.resources as pkg_resources
from pathlib import Path
import numpy as np
import logging

from qcatch import templates
from qcatch.utils import QuantInput, get_input
from qcatch.plots_tables import show_quant_log_table
from qcatch.convert_to_html import create_plotly_plots, modify_html_with_plots
from qcatch.find_retained_cells.run_cell_calling import run_cell_calling

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qcatch")
except PackageNotFoundError:
    __version__ = "unknown"

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
        help="Path to either the .h5ad file itself or to the directory containing the quantification output files."
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        help="Path to the desired output directory (optional) . If provided, QCatch will save all result files and the QC report to this directory without modifying the original input. If not provided, QCatch will overwrite the original `.h5ad` file in place by appending new columns on anndata.obs(if input is a `.h5ad`), or save results in the input directory (if input is a folder of quantification results)."
    )
    parser.add_argument(
        '--chemistry', '-c', 
        type=str, 
        help="Specifies the chemistry used in the experiment, which determines the range for the empty_drops step. Options: '10X_3p_v2', '10X_3p_v3', '10X_3p_v4', '10X_5p_v3', '10X_3p_LT', '10X_HT'. If not provided, we'll use the default range (which is the range used for '10X_3p_v2' and '10X_3p_v3')."
    )
    parser.add_argument(
        '--save_filtered_h5ad', '-s',
        action='store_true',
        help="If enabled with an h5ad input, `qcatch` will save a separate `.h5ad` file containing only the retained cells."
    )
    
    parser.add_argument(
        '--gene_id2name_file', '-g', 
        type=Path,
        default=None,
        help="File provides a mapping from gene IDs to gene names. The file must be a TSV containing two columns—‘gene_id’ (e.g., ENSG00000284733) and ‘gene_name’ (e.g., OR4F29)—without a header row. If not provided, the program will attempt to retrieve the mapping from a remote registry. If that lookup fails, mitochondria plots will not be displayed."
    )
    parser.add_argument(
        '--n_partitions', '-n', 
        type=int, 
        default=None,
        help="Number of partitions (max number of barcodes to consider for ambient estimation). Skip this step if you already specify the chemistry. Otherwise, you can specify the desired `n_partitions`. "
    )
    parser.add_argument(
        '--skip_umap_tsne', '-u',
        action='store_true',
        help='If provided, skips generation of UMAP and t-SNE plots.'
    )
    parser.add_argument(
        '--verbose', '-b',
        action='store_true', 
        help='Enable verbose logging with debug-level messages')
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f"qcatch version {__version__}",
        help='Display the installed version of qcatch.'
    )
    args = parser.parse_args()

    # Set logging level based on the verbose flag
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s :\n %(message)s"
    )
    
    output_dir = args.output
    if output_dir:
        output_dir = Path(output_dir)
        os.makedirs(os.path.join(output_dir), exist_ok=True)
    else:
        # If no output directory is specified, use the input directory/input file's parent directory
        output_dir = Path(args.input.dir)
        
    # Suppress Numba’s debug messages by raising its level to WARNING
    logging.getLogger('numba').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
   
    # add gene_id_2_name if we don't yet have it
    args.input.add_geneid_2_name_if_absent(args.gene_id2name_file, output_dir)
    
    version = __version__
    
    # **** only for development and testing *****
    save_for_quick_test = False # if True, will save the non_ambient_result.pkl file for quick test
    quick_test_mode = True # If True, will skip the cell calling step2
    
    # Run the cell calling process. We will either modify the input file(change the args.input) or save the results in the output directory
    valid_bcs = run_cell_calling(args, output_dir, version, save_for_quick_test, quick_test_mode)

    logger.info("🎨 Generating plots and tables...")
    
    # plots and log, summary tables
    plot_text_elements = create_plotly_plots(args, valid_bcs)
    
    table_htmls = show_quant_log_table(args.input.quant_json_data, args.input.permit_list_json_data)
    
    # Modify HTML with plots
    modify_html_with_plots(
        load_template(),
        os.path.join(output_dir, f'QCatch_report.html'),
        plot_text_elements,
        table_htmls,
        args.input.usa_mode
    )

if __name__ == "__main__":
    main()
