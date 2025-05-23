import logging
from argparse import Namespace
from pathlib import Path

from qcatch import __version__
from qcatch.convert_to_html import create_plotly_plots
from qcatch.find_retained_cells.run_cell_calling import run_cell_calling
from qcatch.logger import generate_warning_html, setup_logger
from qcatch.plots_tables import show_quant_log_table
from qcatch.utils import get_input


def run_qcatch_api(
    input_path: str | Path,
    output: str | Path = None,
    chemistry: str = None,
    gene_id2name_file: str | Path | None = None,
    valid_cell_list: str | Path | None = None,
    skip_umap_tsne: bool = False,
    export_summary_table: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """
    Run QCatch as a Python API.

    Args:
        input_path: Path to .h5ad file or quant folder
        chemistry: Chemistry type (same as CLI)
        gene_id2name_file: TSV with gene ID to name map
        valid_cell_list: Optional list of valid barcodes
        skip_umap_tsne: Skip clustering plots
        export_summary_table: Whether to compute and return summary table
        logger: Optional custom logger

    Returns
    -------
        dict with:
            - 'anndata': updated AnnData object
            - 'valid_barcodes': list of barcodes
            - 'figures': list of plotly figures
            - 'summary_table_html': str (optional)
            - 'warning_html': str
    """
    logger = logger or setup_logger("qcatch", verbose=False)

    # Convert and validate input
    input = get_input(str(input_path))

    output_dir = Path(output) if output else Path(input.dir)
    output_dir.mkdir(exist_ok=True)

    input.add_geneid_2_name_if_absent(gene_id2name_file, output_dir)
    version = f"{__version__}-API"

    # Run cell calling
    save_for_quick_test = False
    quick_test_mode = False
    args = Namespace(
        input=input,
        chemistry=chemistry,
        valid_cell_list=valid_cell_list,
        output=output_dir,
        n_partitions=None,
        verbose=False,
    )
    valid_barcodes = run_cell_calling(
        args=args,
        version=version,
        save_for_quick_test=save_for_quick_test,
        quick_test_mode=quick_test_mode,
    )

    if len(valid_barcodes) == 0:
        raise ValueError("❗ No valid barcodes found. Cell calling failed.")

    # Create plots and summary tables
    plot_args = Namespace(
        input=input,
        output=output_dir,
        skip_umap_tsne=skip_umap_tsne,
    )
    plot_texts, _ = create_plotly_plots(
        args=plot_args,
        valid_bcs=valid_barcodes,
    )

    summary_html = ""
    if export_summary_table:
        summary_html = show_quant_log_table(input.quant_json_data, input.permit_list_json_data)

    warning_html = generate_warning_html(logger.get_record_log())

    return {
        "anndata": input.mtx_data,
        "valid_barcodes": valid_barcodes,
        "figures": plot_texts,  # HTML strings or optionally real plotly.Figure
        "summary_table_html": summary_html,
        "warning_html": warning_html,
    }
