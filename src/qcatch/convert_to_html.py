import logging
import os
from argparse import Namespace

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from qcatch.logger import QCatchLogger

# from qcatch.input_processing import
from qcatch.plots_tables import (
    barcode_frequency_plots,
    create_plots_from_embedding,
    generate_embeddings,
    generate_gene_histogram,
    generate_knee_plots,
    generate_seq_saturation,
    generate_SUA_plots,
    generate_summary_table,
    mitochondria_plot,
    umap_tsne_plot,
    umi_dedup,
)

logger = logging.getLogger("qcatch")
assert isinstance(logger, QCatchLogger), "Logger is not a QCatchLogger. Call setup_logger() in main.py first."


def generate_pipeline_summary_html(pipeline_info: dict | None) -> str:
    """
    Generate HTML for the cell filtering pipeline summary section.

    Parameters
    ----------
    pipeline_info
        Dictionary containing pipeline metadata with keys:
        - cell_calling_method: 'user_provided' or 'internal'
        - initial_cells: int
        - doublet_removal_enabled: bool
        - n_doublets_removed: int or None
        - n_singlets_retained: int or None
        - final_retained_cells: int

    Returns
    -------
    str
        HTML string for the pipeline summary alert box
    """
    if not pipeline_info:
        return ""

    # Step 1: Cell Calling
    if pipeline_info["cell_calling_method"] == "user_provided":
        step1_html = f"""
          <div class="text-center">
            <div class="mb-1"><strong style="font-size: 0.95rem;">🧬 Step 1: Cell Calling</strong></div>
            <div style="font-size: 0.85rem;">User-provided cell list</div>
            <div class="fw-bold text-primary" style="font-size: 1.1rem;">{pipeline_info["initial_cells"]:,} cells</div>
          </div>
        """
    else:  # internal
        step1_html = f"""
          <div class="text-center">
            <div class="mb-1"><strong style="font-size: 0.95rem;">🧬 Step 1: Cell Calling</strong></div>
            <div style="font-size: 0.85rem;">Internal 2-step filtering<br>(OrdMag + EmptyDrops)</div>
            <div class="fw-bold text-primary" style="font-size: 1.1rem;">{pipeline_info["initial_cells"]:,} cells</div>
          </div>
        """

    # Step 2: Doublet Removal
    if pipeline_info["doublet_removal_enabled"] and pipeline_info["n_doublets_removed"] is not None:
        step2_html = f"""
          <div class="text-center">
            <div class="mb-1"><strong style="font-size: 0.95rem;">🔍 Step 2: Doublet Removal</strong></div>
            <div class="text-danger" style="font-size: 0.85rem;">- {pipeline_info["n_doublets_removed"]:,} doublets</div>
            <div class="fw-bold text-primary" style="font-size: 1.1rem;">{pipeline_info["n_singlets_retained"]:,} singlets</div>
          </div>
        """
    else:
        step2_html = """
          <div class="text-center">
            <div class="mb-1"><strong style="font-size: 0.95rem;">⊝ Step 2: Doublet Removal</strong></div>
            <div class="text-muted" style="font-size: 0.85rem;">Not applied</div>
            <div class="fw-bold text-muted" style="font-size: 1.1rem;">—</div>
          </div>
        """

    # Final result
    final_html = f"""
          <div class="text-center">
            <div class="mb-1"><strong style="font-size: 0.95rem;">✨ Final Result</strong></div>
            <div style="font-size: 0.85rem;">Retained cells</div>
            <div class="fw-bold text-primary" style="font-size: 1.3rem;">{pipeline_info["final_retained_cells"]:,}</div>
          </div>
    """

    # Combine all parts in horizontal layout (3 columns)
    full_html = f"""
    <div class="alert alert-info my-2 py-2 px-3" role="alert">
      <h6 class="text-center mb-2 pb-2 border-bottom fw-bold" style="font-size: 1.05rem;">🔬 Cell Filtering Pipeline</h6>
      <div class="row g-0">
        <div class="col-md-4 border-end px-2">
          {step1_html}
        </div>
        <div class="col-md-4 border-end px-2">
          {step2_html}
        </div>
        <div class="col-md-4 px-2">
          {final_html}
        </div>
      </div>
    </div>
    """

    return full_html


def create_plotly_plots(
    args: Namespace,
    valid_bcs: list[str],
    bcs_with_doublets: list[str] | None = None,
    pipeline_info: dict | None = None,
) -> tuple[dict[str, str], str, str]:
    """
    Generate interactive Plotly plots and summary tables from Alevin-fry quantification data.

    Parameters
    ----------
        args: An object containing input data and parameters including feature dump data, map JSON data,
              mtx data (AnnData object), usa_mode flag, and skip_umap_tsne flag.
        valid_bcs: A list or set of valid barcodes to filter cells (singlets after doublet removal).
        bcs_with_doublets: Optional list of barcodes before doublet removal (includes both singlets and doublets).
                          Used when visualize_doublets flag is enabled.

    Returns
    -------
        tuple: A tuple containing:
            - dict: A dictionary of Plotly plot HTML div strings keyed by plot identifiers.
            - str: An HTML string representing the summary table.
    """
    feature_dump_data = args.input.feature_dump_data
    map_json_data = args.input.map_json_data
    adata = args.input.mtx_data
    usa_mode = args.input.usa_mode

    # Filter cells with zero reads from featureDump data
    data = feature_dump_data[
        (feature_dump_data["deduplicated_reads"] >= 1) & (feature_dump_data["num_genes_expressed"] >= 1)
    ]

    retained_data = data[data["barcodes"].isin(valid_bcs)]

    # Sort by "deduplicated_reads" and assign rank
    data = data.sort_values("deduplicated_reads", ascending=False).reset_index(drop=True)
    data["rank"] = data.index + 1  # 1-based rank

    # get filtered adata
    filtered_mask = adata.obs["barcodes"].isin(valid_bcs) if args.input.is_h5ad else adata.obs_names.isin(valid_bcs)
    # NOTE: safe but maybe time consuming
    filtered_adata = adata[filtered_mask, :].copy()

    # ---------------- Tab1 - Knee Plots ---------------
    fig_knee_1, fig_knee_2 = generate_knee_plots(data, valid_bcs)

    # ---------------- Generate summary table content ----------------
    if usa_mode:
        # add up all layers
        all_mtx_filtered = (
            filtered_adata.layers["spliced"] + filtered_adata.layers["unspliced"] + filtered_adata.layers["ambiguous"]
        )
    else:
        all_mtx_filtered = filtered_adata.X
    # Get total detected genes for reatined cells
    total_detected_genes = (all_mtx_filtered > 0).sum(axis=0)
    # Count genes with at least one UMI
    total_detected_genes = np.count_nonzero(total_detected_genes)
    # calculate median genes per cell
    all_mtx_gene_per_cell = (all_mtx_filtered > 0).sum(axis=1).tolist()
    median_genes_per_cell = int(np.median(all_mtx_gene_per_cell))

    # get mapping rate
    mapping_rate = None
    if map_json_data:
        num_processed = map_json_data.get("num_processed") or map_json_data["num_reads"]
        if num_processed and map_json_data.get("num_mapped"):
            mapping_rate = round(map_json_data["num_mapped"] / num_processed * 100, 2)
            if mapping_rate < 60:  # less than 60% mapping rate
                msg = "High-quality datasets typically exhibit a mapping rate of around 90%. In this dataset, the mapping rate is below 60%, which may indicate a mismatch between the sample and the reference -- often due to sample contamination or using an inappropriate reference transcriptome."
                logger.record_warning(msg)
            elif mapping_rate < 80:  # less than 80% mapping rate
                msg = "High-quality datasets typically exhibit a mapping rate of around 90%. In this dataset, the mapping rate is below 80%, which may indicate low data quality."
                logger.record_warning(msg)

    seq_saturation_value = generate_seq_saturation(retained_data)

    export_summary_table = args.export_summary_table
    summary_table_html, summary = generate_summary_table(
        data,
        valid_bcs,
        total_detected_genes,
        median_genes_per_cell,
        mapping_rate,
        seq_saturation_value,
    )
    # Optionally: Save the summary table to a CSV file
    if export_summary_table:
        summary_table_file_path = os.path.join(args.output, "summary_table.csv")
        logger.info(f"🍡 Saving summary table to CSV at: {summary_table_file_path}")
        # Save the summary dictionary as a single-row CSV file with keys as column headers
        pd.DataFrame([summary]).to_csv(summary_table_file_path, index=False)

    # ---------------- Tab2 - Barcode Frequency Plots ---------------
    fig_bc_freq_all_plots = barcode_frequency_plots(data, valid_bcs)

    # ---------------- Tab3 - Collapsing ---------------
    # NOTE: use the retained data for umi_collapse plot
    fig_umi_dedup, mean_dedup_rate = umi_dedup(retained_data)

    # ---------------- Tab4 - Histogram of Genes Detected ---------------
    fig_hist_genes = generate_gene_histogram(data, is_all_cells=True)
    fig_hist_genes_filtered = generate_gene_histogram(retained_data, is_all_cells=False)
    # if our adata object doesn't already contain the gene symbol, then skip the mitochondrial plot.
    if "gene_symbol" in adata.var.columns:
        fig_mt = mitochondria_plot(adata, is_all_cells=True)
        fig_mt_filtered = mitochondria_plot(filtered_adata, is_all_cells=False)
    else:
        fig_mt = None
        fig_mt_filtered = None

    # ---------------- Tab5 - SUA plots ---------------
    if usa_mode:
        fig_SUA_bar_html, fig_S_ratio_html = generate_SUA_plots(adata, is_all_cells=True)
        fig_SUA_bar_filtered_html, fig_S_ratio_filtered_html = generate_SUA_plots(filtered_adata, is_all_cells=False)
    # ---------------- Tab6 - UMAP ---------------
    if not args.skip_umap_tsne:
        if bcs_with_doublets is not None and args.visualize_doublets:
            # ===== SHARED EMBEDDING MODE =====
            logger.info("🔬 Generating shared embeddings for doublet visualization...")
            # Use full dataset with doublets for embedding generation
            # Use "barcodes" column (same as remove_doublets) instead of obs_names
            doublet_mask = adata.obs["barcodes"].isin(bcs_with_doublets)
            adata_with_doublets = adata[doublet_mask, :].copy()

            # Count actual doublets from data (accounting for NA values)
            n_doublets = int(adata_with_doublets.obs["predicted_doublet"].sum())
            n_singlets = int((~adata_with_doublets.obs["predicted_doublet"]).sum())
            n_na = int(adata_with_doublets.obs["predicted_doublet"].isna().sum())

            if n_na > 0:
                logger.warning(f"⚠️ Found {n_na} cells with NA doublet status (not evaluated by Scrublet)")
            logger.info(
                f"📊 Processing {adata_with_doublets.n_obs} cells ({n_singlets} singlets + {n_doublets} doublets{f' + {n_na} NA' if n_na > 0 else ''})..."
            )

            # Generate embeddings ONCE on all cells (including doublets)
            adata_embedded = generate_embeddings(adata_with_doublets, run_clustering=True)
            logger.info("✅ Embeddings generated. Creating plots...")

            # View 1: "Retained Cells" - filter to singlets only, same coordinates
            # Convert to numpy array to avoid pandas BooleanArray indexing issues
            singlet_mask = ~adata_embedded.obs["predicted_doublet"].fillna(True).to_numpy()
            # Use view instead of copy for better performance
            adata_singlets_view = adata_embedded[singlet_mask, :]
            fig_umap, fig_tsne = create_plots_from_embedding(
                adata_singlets_view, color_by="leiden", plot_label="(Retained Cells Only)"
            )

            # View 2: "With Doublets" - all cells, same coordinates
            fig_umap_doublets, fig_tsne_doublets = create_plots_from_embedding(
                adata_embedded, color_by="doublet_status", plot_label="(With Doublets)"
            )

            # Code text for shared embedding mode
            code_text = """
        # Shared embedding for doublet visualization
        # Preprocessing on all cells (singlets + doublets)
        sc.pp.normalize_total(adata_with_doublets)
        sc.pp.log1p(adata_with_doublets)
        sc.pp.highly_variable_genes(adata_with_doublets, n_top_genes=min(2000, n_valid))
        sc.tl.pca(adata_with_doublets)
        sc.pp.neighbors(adata_with_doublets)
        sc.tl.umap(adata_with_doublets)
        sc.tl.tsne(adata_with_doublets)
        sc.tl.leiden(adata_with_doublets, flavor="igraph", n_iterations=2)

        # View 1: "Retained Cells Only" - Filter to singlets, color by Leiden clusters
        singlet_mask = ~adata_with_doublets.obs["predicted_doublet"].fillna(True)
        adata_singlets = adata_with_doublets[singlet_mask, :]

        umap_df_singlets = pd.DataFrame(adata_singlets.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        umap_df_singlets["leiden"] = adata_singlets.obs["leiden"].values
        fig_umap_singlets = px.scatter(umap_df_singlets, x="UMAP1", y="UMAP2", color="leiden",
                                        title="UMAP with Leiden Clusters (Retained Cells Only)")

        tsne_df_singlets = pd.DataFrame(adata_singlets.obsm["X_tsne"], columns=["TSNE1", "TSNE2"])
        tsne_df_singlets["leiden"] = adata_singlets.obs["leiden"].values
        fig_tsne_singlets = px.scatter(tsne_df_singlets, x="TSNE1", y="TSNE2", color="leiden",
                                        title="t-SNE with Leiden Clusters (Retained Cells Only)")

        # View 2: "With Doublets" - All cells, color by doublet status
        adata_with_doublets.obs["doublet_label"] = adata_with_doublets.obs["predicted_doublet"].map(
            {True: "Doublet", False: "Singlet"}).astype(str)

        umap_df_all = pd.DataFrame(adata_with_doublets.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        umap_df_all["doublet_label"] = adata_with_doublets.obs["doublet_label"].values
        umap_df_all["doublet_score"] = adata_with_doublets.obs["doublet_score"].values
        fig_umap_doublets = px.scatter(umap_df_all, x="UMAP1", y="UMAP2", color="doublet_label",
                                        hover_data=["doublet_score"],
                                        color_discrete_map={"Singlet": "#3498db", "Doublet": "#e74c3c"},
                                        title="UMAP with Doublet Classification (With Doublets)")

        tsne_df_all = pd.DataFrame(adata_with_doublets.obsm["X_tsne"], columns=["TSNE1", "TSNE2"])
        tsne_df_all["doublet_label"] = adata_with_doublets.obs["doublet_label"].values
        tsne_df_all["doublet_score"] = adata_with_doublets.obs["doublet_score"].values
        fig_tsne_doublets = px.scatter(tsne_df_all, x="TSNE1", y="TSNE2", color="doublet_label",
                                        hover_data=["doublet_score"],
                                        color_discrete_map={"Singlet": "#3498db", "Doublet": "#e74c3c"},
                                        title="t-SNE with Doublet Classification (With Doublets)")
        """
        else:
            # ===== STANDARD MODE (current behavior) =====
            # Only use singlets for embedding
            fig_umap, fig_tsne, code_text = umap_tsne_plot(filtered_adata)
            fig_umap_doublets = None
            fig_tsne_doublets = None

    else:
        fig_umap = fig_tsne = None
        fig_umap_doublets = fig_tsne_doublets = None
        code_text = ""
        logger.info("🦦 Skipping UMAP and t-SNE plots as per user request.")

    # Convert plots to HTML div strings
    plots = {
        # ----tab1----(
        "knee_plot1-1": fig_knee_1.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        ),
        "knee_plot1-2": fig_knee_2.to_html(full_html=False, include_plotlyjs="cdn"),
        # ----tab2----
        "bc_freq_all_plots": fig_bc_freq_all_plots.to_html(full_html=False, include_plotlyjs="cdn"),
        # ----tab3----
        "hist_gene3-1": fig_hist_genes.to_html(full_html=False, include_plotlyjs="cdn"),
        # filtered data
        "hist_gene_filtered_3-1": fig_hist_genes_filtered.to_html(full_html=False, include_plotlyjs="cdn"),
        # ----tab4----
        "umi_dedup4": fig_umi_dedup.to_html(full_html=False, include_plotlyjs="cdn"),
        # ---tab6----
    }
    if fig_umap and fig_tsne:
        # Standard plots (retained cells only)
        plots["umap_plot_filtered_6-1"] = fig_umap.to_html(full_html=False, include_plotlyjs="cdn")
        plots["tsne_plot_filtered_6-2"] = fig_tsne.to_html(full_html=False, include_plotlyjs="cdn")

        # Doublet plots (if available)
        if fig_umap_doublets and fig_tsne_doublets:
            plots["umap_plot_doublets_6-1"] = fig_umap_doublets.to_html(full_html=False, include_plotlyjs="cdn")
            plots["tsne_plot_doublets_6-2"] = fig_tsne_doublets.to_html(full_html=False, include_plotlyjs="cdn")
    elif args.skip_umap_tsne:
        # If UMAP/t-SNE is skipped, add skip message
        skip_message = """
        <div class="alert alert-warning text-center my-4" role="alert">
            <i class="bi bi-info-circle-fill me-2"></i>
            <strong>UMAP and t-SNE plots skipped</strong>
            <p class="mb-0 mt-2 small">Clustering visualization was disabled with <code>--skip_umap_tsne</code> flag.</p>
        </div>
        """
        plots["umap_skip_message"] = skip_message

    if fig_mt or fig_mt_filtered:
        #  2nd plot in tab3
        plots["fig_mt3-2"] = fig_mt.to_html(full_html=False, include_plotlyjs="cdn")
        plots["fig_mt_filtered_3-2"] = fig_mt_filtered.to_html(full_html=False, include_plotlyjs="cdn")

    # add key pair for SUA plots, is usa_mode is True
    if usa_mode:
        # ----tab5----
        plots["SUA_bar5-1"] = fig_SUA_bar_html
        plots["S_ratio5-2"] = fig_S_ratio_html
        plots["SUA_bar_filtered_5-1"] = fig_SUA_bar_filtered_html
        plots["S_ratio_filtered_5-2"] = fig_S_ratio_filtered_html
    plot_text_elements = (plots, summary_table_html)

    # Generate pipeline summary HTML
    pipeline_summary_html = generate_pipeline_summary_html(pipeline_info)

    return plot_text_elements, code_text, pipeline_summary_html


def modify_html_with_plots(
    soup: BeautifulSoup,
    output_html_path: str,
    plot_text_elements: tuple[dict[str, str], str],
    table_htmls: tuple[str, str],
    code_texts: str,
    warning_html: str,
    usa_mode: bool,
    pipeline_summary_html: str = "",
) -> None:
    """
    Modify an existing HTML document by inserting Plotly plots, updating summary and log info tables, optionally removing SUA tab content, and adding warning messages.

    Parameters
    ----------
    soup
        Parsed HTML document as a BeautifulSoup object.
    output_html_path
        Path to save the modified HTML file.
    plot_text_elements
        Tuple containing:
        - A dictionary of Plotly plot HTML div strings keyed by plot identifiers.
        - An HTML string representing the summary table.
    table_htmls
        Tuple containing HTML strings for the quant log and permit list log info tables.
    code_texts
        Python code snippet to insert into the report.
    warning_html
        HTML string containing warning messages.
    usa_mode
        Flag indicating whether SUA mode is enabled; if False, SUA tab is removed.

    Returns
    -------
        None
    """
    plots, summary_table_html = plot_text_elements
    quant_json_table_html, permit_list_table_html = table_htmls

    updated_sections = []
    missing_sections = []

    # Updating plots
    for plot_id, plot_html in plots.items():
        plot_div = soup.find(id=plot_id)
        if plot_div:
            plot_div.clear()
            plot_div.append(BeautifulSoup(plot_html, "html.parser"))
            updated_sections.append(plot_id)
        else:
            missing_sections.append(f"Plot '{plot_id}'")

    # Summary and log info tables
    for table_id, table_html in [
        ("summary_tbody", summary_table_html),
        ("table-body-quant-log-info", quant_json_table_html),
        ("table-body-permit-log-info", permit_list_table_html),
    ]:
        table_body = soup.find(id=table_id)
        if table_body:
            table_body.clear()
            table_body.append(BeautifulSoup(table_html, "html.parser"))
            updated_sections.append(table_id)
        else:
            missing_sections.append(f"Table '{table_id}'")
    # Remove SUA tab if usa_mode is False
    if not usa_mode:
        sua_tab = soup.find(id="tab5_nav")  # Navbar tab
        sua_content = soup.find(id="tab5")  # Content section
        if sua_tab:
            sua_tab.decompose()
        if sua_content:
            sua_content.decompose()

    # Add the code block for plots help text

    # create tags: <pre><code class="language-python">...</code></pre>
    code_outer = soup.new_tag("pre", **{"class": "scroll-box"})
    code_inner = soup.new_tag("code", **{"class": "language-python"})
    code_inner.string = code_texts
    code_outer.append(code_inner)

    # insert to <div id="code-block">
    code_block = soup.find("div", id="code-block-umap")
    code_block.clear()  # Optional: clear previous content
    code_block.append(code_outer)

    # Add warning section
    if warning_html.strip():
        warning_section = soup.find(id="qc-warning-container")
        if warning_section:
            warning_section.clear()
            warning_section.append(BeautifulSoup(warning_html, "html.parser"))
            updated_sections.append("qc-warning-container")
        else:
            missing_sections.append("Warning section with id='qc-warning-container'")

    # Add pipeline summary section
    if pipeline_summary_html and pipeline_summary_html.strip():
        pipeline_container = soup.find("div", id="pipeline-summary-container")
        if pipeline_container:
            pipeline_container.clear()
            pipeline_container.append(BeautifulSoup(pipeline_summary_html, "html.parser"))
            updated_sections.append("pipeline-summary-container")
        else:
            missing_sections.append("Pipeline summary section with id='pipeline-summary-container'")

    # Write the updated HTML to a new file
    with open(output_html_path, "w", encoding="utf-8") as file:
        file.write(str(soup))

    if missing_sections:
        for missing in missing_sections:
            logger.warning(f"⚠️ {missing} not found in the HTML.")
    if len(missing_sections) == 0:
        logger.info(f"🚀 QC report successfully updated all elements and saved to: {output_html_path}")
    elif len(updated_sections) != 0:
        logger.info(f"🚀 QC report successfully updated and saved to: {output_html_path}")
