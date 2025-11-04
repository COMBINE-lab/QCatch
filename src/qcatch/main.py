import argparse
import importlib.resources as pkg_resources
import os
from pathlib import Path

from bs4 import BeautifulSoup

from qcatch import __version__, templates
from qcatch.convert_to_html import create_plotly_plots, modify_html_with_plots
from qcatch.find_retained_cells.run_cell_calling import run_cell_calling
from qcatch.logger import generate_warning_html, setup_logger
from qcatch.plots_tables import show_quant_log_table
from qcatch.utils import get_input


def load_template():
    """
    Load and parse the HTML report template using BeautifulSoup.

    Returns
    -------
    soup
        Parsed HTML template as a BeautifulSoup object.
    """
    template_path = pkg_resources.files(templates) / "report_template.html"
    with open(template_path, encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
    return soup


def main():
    """
    Main entry point for the QCatch command-line interface.

    This function sets up the argument parser, processes the input arguments, and orchestrates the
    execution of the QCatch pipeline, including cell calling, plot generation, and report creation.
    """
    parser = argparse.ArgumentParser(description="QCatch: Command-line Interface")
    # Add command-line arguments
    parser.add_argument(
        "--input",
        "-i",
        type=get_input,
        required=True,
        help="Path to either the .h5ad file itself or to the directory containing the quantification output files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the desired output directory (optional) . If provided, QCatch will save all result files and the QC report to this directory without modifying the original input. If not provided, QCatch will overwrite the original `.h5ad` file in place by appending new columns on anndata.obs(if input is a `.h5ad`), or save results in the input directory (if input is a folder of quantification results).",
    )
    parser.add_argument(
        "--chemistry",
        "-c",
        type=str,
        help="Specifies the chemistry used in the experiment, which determines the range for the empty_drops step. Supported options: '10X_3p_v2', '10X_3p_v3', '10X_3p_v4', '10X_5p_v3', '10X_3p_LT', '10X_HT'. If you used a standard 10X chemistry (e.g., '10X_3p_v2', '10X_3p_v3') and performed quantification with `simpleaf`(v0.19.5 or later), QCatch can usually infer the correct chemistry automatically from the metadata. If inference fails, QCatch will stop and prompt you to provide the chemistry explicitly via this flag. ",
    )
    parser.add_argument(
        "--save_filtered_h5ad",
        "-s",
        action="store_true",
        help="If enabled with an h5ad input, QCatch will save a separate `.h5ad` file containing only the retained cells.",
    )

    parser.add_argument(
        "--gene_id2name_file",
        "-g",
        type=Path,
        default=None,
        help="File provides a mapping from gene IDs to gene names. The file must be a TSV containing two columns—‘gene_id’ (e.g., ENSG00000284733) and ‘gene_name’ (e.g., OR4F29)—without a header row. If not provided, the program will attempt to retrieve the mapping from a remote registry. If that lookup fails, mitochondria plots will not be displayed.",
    )
    parser.add_argument(
        "--valid_cell_list",
        "-l",
        type=Path,
        default=None,
        help="File provides a user-specified list of valid cell barcode. The file must be a TSV containing one column with cell barcodes without a header row. If provided, qcatch will skip the internal cell calling steps and and use the supplied list instead.",
    )
    parser.add_argument(
        "--n_partitions",
        "-n",
        type=int,
        default=None,
        help="Number of partitions (max number of barcodes to consider for ambient estimation). Use `--n_partitions` only when working with a custom or unsupported chemistry. When provided, this value will override the chemistry-based configuration during the cell-calling step.",
    )
    parser.add_argument(
        "--skip_umap_tsne", "-u", action="store_true", help="If provided, skips generation of UMAP and t-SNE plots."
    )
    parser.add_argument(
        "--export_summary_table",
        "-e",
        action="store_true",
        help="If enabled, QCatch will export the summary metrics as a separate CSV file.",
    )
    parser.add_argument("--verbose", "-b", action="store_true", help="Enable verbose logging with debug-level messages")

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"qcatch version {__version__}",
        help="Display the installed version of qcatch.",
    )
    args = parser.parse_args()

    logger = setup_logger("qcatch", args.verbose)

    # If no output directory is specified, use the input directory/input file's parent directory
    args.output = Path(args.output) if args.output else Path(args.input.dir)
    os.makedirs(args.output, exist_ok=True)
    # add gene_id_2_name if we don't yet have it
    args.input.add_geneid_2_name_if_absent(args.gene_id2name_file, args.output)

    version = __version__

    # ****  ------------------------------- *****
    # **** only for development and testing *****
    save_for_quick_test = False  # if True, will save the non_ambient_result.pkl file for quick test
    quick_test_mode = False  # If True, will skip the cell calling step2
    # ****  ------------------------------- *****

    # Run the cell calling process. We will either modify the input file(change the args.input) or save the results in the output directory
    valid_bcs = run_cell_calling(args, version, save_for_quick_test, quick_test_mode)
    logger.info("🎨 Generating plots and tables...")
    if len(valid_bcs) == 0:
        msg = "❗️ Error: No valid barcodes found. Skip QC report HTML generation."
        logger.error(msg)
        return
    # plots and log, summary tables
    plot_text_elements, code_texts = create_plotly_plots(args, valid_bcs)

    table_htmls = show_quant_log_table(args.input.quant_json_data, args.input.permit_list_json_data)

    warning_html = generate_warning_html(logger.get_record_log())

    # Modify HTML with plots
    modify_html_with_plots(
        load_template(),
        os.path.join(args.output, "QCatch_report.html"),
        plot_text_elements,
        table_htmls,
        code_texts,
        warning_html,
        args.input.usa_mode,
    )


if __name__ == "__main__":
    main()
