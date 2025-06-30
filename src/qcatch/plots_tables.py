from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scanpy as sc
from plotly.subplots import make_subplots

from qcatch.logger import QCatchLogger

logger = logging.getLogger(__name__)
assert isinstance(logger, QCatchLogger), "Logger is not a QCatchLogger. Call setup_logger() in main.py first."


def apply_uniform_style(fig: go.Figure) -> go.Figure:
    """Apply a consistent style with a solid background matching the container to a Plotly figure."""
    fig.update_layout(
        title_x=0.5,
        margin={"t": 40, "l": 30, "r": 30, "b": 30},
        plot_bgcolor="rgba(0,0,0,0)",  # Fully transparent background
        paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor="#fff",  # Solid white background for plot area
        # paper_bgcolor="#fff",  # Solid white background for entire figure
        font={"family": "Arial, sans-serif", "size": 14, "color": "#2c3e50"},
        title_font={"size": 16, "color": "#34495e"},
        xaxis={
            "showgrid": True,
            "gridcolor": "lightgrey",
            "linecolor": "#34495e",
            "tickfont": {"size": 14, "color": "#34495e"},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "lightgrey",
            "linecolor": "#34495e",
            "tickfont": {"size": 14, "color": "#34495e"},
        },
    )
    return fig


def get_cell_label(is_all_cells: bool) -> str:
    """Returns a string label indicating whether the data represents all cells or only retained cells."""
    return " (All Cells)" if is_all_cells else " (Retained Cells Only)"


# set up width and height for the plots in horizontal
width = 480
height = 360
opacity = 0.5


def generate_knee_plots(data: pd.DataFrame, valid_bcs: list[str]) -> tuple[go.Figure, go.Figure]:
    """
    Generate knee plots for UMI counts and genes detected versus cell rank, colored by cell type.

    Parameters
    ----------
    data
        Feature dump data.
    valid_bcs
        List of valid barcodes.

    Returns
    -------
    Two Plotly figures for UMI counts and detected genes.
    """
    width = 520
    # Map is_retained to human-readable cell_type
    data["cell_type"] = data["barcodes"].isin(valid_bcs).map({True: "Retained Cell", False: "Background"})

    # 1.1 Knee Plot 1: Rank vs UMI Counts, colored by cell_type
    fig_knee_1 = px.scatter(
        data,
        x="rank",
        y="deduplicated_reads",
        log_x=True,
        log_y=True,
        title="UMI Counts vs Cell Rank (All Cells)",
        labels={"rank": "Cell Rank", "deduplicated_reads": "UMI Counts", "cell_type": "Cell Type"},
        color="cell_type",
        color_discrete_map={"Retained Cell": "#636EFA", "Background": "lightgrey"},
        width=width,
        height=height,
        opacity=opacity,
    )

    # 1.2 Knee Plot 2: Rank vs Genes Detected, colored by cell_type
    fig_knee_2 = px.scatter(
        data,
        x="rank",
        y="num_genes_expressed",
        log_x=True,
        log_y=True,
        title="Genes Detected against Cell Rank (All Cells)",
        labels={"rank": "Cell Rank", "num_genes_expressed": "Number of Detected Genes", "cell_type": "Cell Type"},
        color="cell_type",
        color_discrete_map={"Retained Cell": "#636EFA", "Background": "lightgrey"},
        width=width,
        height=height,
        opacity=opacity,
    )
    fig_knee_2.update_yaxes(range=[0, np.log10(data["num_genes_expressed"].max()) + 0.1])
    # modify the legend
    fig_knee_1.update_layout(legend_title_text="", title_x=0.5)

    fig_knee_2.update_layout(legend_title_text="", title_x=0.5)

    fig_knee_1.update_traces(marker={"size": 5})
    fig_knee_2.update_traces(marker={"size": 5})

    return apply_uniform_style(fig_knee_1), apply_uniform_style(fig_knee_2)


def barcode_frequency_plots(data: pd.DataFrame, valid_bcs: list[str]) -> go.Figure:
    """
    Generate subplots showing barcode frequency vs UMI counts, barcode frequency vs detected genes, and UMI counts vs detected genes.

    Parameters
    ----------
    data
        Barcode and cell metrics.
    valid_bcs
        Valid barcode strings.

    Returns
    -------
    A Plotly Figure with three subplots.
    """
    width = 1200
    height = 400
    opacity = 0.5
    # Map is_retained to human-readable cell_type
    data["cell_type"] = np.where(data["barcodes"].isin(valid_bcs), "Retained Cell", "Background")

    # Create subplots: 1 row, 3 columns
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Barcode Frequency vs UMI Counts (All Cells)",
            "Barcode Frequency vs Detected Genes (All Cells)",
            "UMI Counts vs Detected Genes (All Cells)",
        ],
        shared_yaxes=False,
    )

    # Plot 1 (legend enabled)
    for trace in px.scatter(
        data,
        x="corrected_reads",
        y="deduplicated_reads",
        color="cell_type",
        color_discrete_map={"Retained Cell": "#636EFA", "Background": "lightgrey"},
        opacity=opacity,
    ).data:
        fig.add_trace(trace, row=1, col=1)

    # Plot 2 (legend disabled)
    for trace in px.scatter(
        data,
        x="corrected_reads",
        y="num_genes_expressed",
        color="cell_type",
        color_discrete_map={"Retained Cell": "#636EFA", "Background": "lightgrey"},
        opacity=opacity,
    ).data:
        trace.showlegend = False  # Disable legend for this trace
        fig.add_trace(trace, row=1, col=2)

    # Plot 3 (legend disabled)
    for trace in px.scatter(
        data,
        x="deduplicated_reads",
        y="num_genes_expressed",
        color="cell_type",
        color_discrete_map={"Retained Cell": "#636EFA", "Background": "lightgrey"},
        opacity=opacity,
    ).data:
        trace.showlegend = False  # Disable legend for this trace
        fig.add_trace(trace, row=1, col=3)

    fig.update_layout(
        width=width,
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3, "xanchor": "center", "x": 0.5, "title_text": ""},
        margin={"t": 60, "b": 80},
        title_x=0.5,
    )
    fig = apply_uniform_style(fig)
    # Explicitly reinforce grid lines for all subplots without overriding other style
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", linecolor="#34495e")
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", linecolor="#34495e")
    # Add axis labels for all three subplots
    fig.update_xaxes(title_text="Barcode frequency", row=1, col=1)
    fig.update_yaxes(title_text="UMI Counts", row=1, col=1)

    fig.update_xaxes(title_text="Barcode frequency", row=1, col=2)
    fig.update_yaxes(title_text="Detected Genes", row=1, col=2)

    fig.update_xaxes(title_text="UMI Counts", row=1, col=3)
    fig.update_yaxes(title_text="Detected Genes", row=1, col=3)
    return fig


def generate_gene_histogram(data: pd.DataFrame, is_all_cells: bool) -> go.Figure:
    """
    Generate a histogram plot of the number of detected genes per cell.

    Parameters
    ----------
    data
        Feature dump data.
    is_all_cells
        True if all cells are included, False for retained cells only.

    Returns
    -------
    A Plotly Figure representing the histogram.
    """
    gene_detected = data["num_genes_expressed"].tolist()
    title_suffix = get_cell_label(is_all_cells)

    # Create a histogram plot
    fig_hist_genes = go.Figure(
        data=[
            go.Histogram(
                x=gene_detected,
                name="Number of detected genes",
                opacity=0.8,
            )
        ],
        layout=go.Layout(
            title="Histogram of detected genes per cell" + title_suffix,
            xaxis={"title": "Number of detected genes", "autorange": True},
            yaxis={"title": "Count", "type": "log", "dtick": 1},
            width=width,
            height=height,
            bargap=0.05,
        ),
    )
    return apply_uniform_style(fig_hist_genes)


def generate_seq_saturation(data: pd.DataFrame) -> float:
    """
    Calculate sequencing saturation percentage for cells.

    Parameters
    ----------
    data
        Feature dump data.

    Returns
    -------
    Sequencing saturation as a percentage.
    """
    # Sequencing Saturation. x = meanReadsPerCell; y = seq saturation
    data = data.sort_values(by="corrected_reads")

    # Get the reads per cell array
    unfiltered_correct_reads_array = data["corrected_reads"].tolist()

    # Calculate Sequencing Saturation (1 - Deduplication Rate)
    dedup_rate_array = data["dedup_rate"].tolist()
    unfiltered_seq_saturation_array = [1 - x for x in dedup_rate_array]

    # Filter out cells with less than 20 genes detected
    filtered_indices = [i for i, val in enumerate(unfiltered_correct_reads_array) if val >= 20]
    # handle case where no cells have 20 genes detected
    if filtered_indices:
        idx_start = filtered_indices[0]
    else:
        idx_start = len(unfiltered_correct_reads_array)

    # correct_reads_array = unfiltered_correct_reads_array[idx_start:]
    seq_saturation_array = unfiltered_seq_saturation_array[idx_start:]

    # Calculate average sequencing saturation value for all filtered cells
    seq_saturation_value = sum(seq_saturation_array) / len(seq_saturation_array) if seq_saturation_array else 0
    seq_saturation_percent = round(seq_saturation_value * 100, 2)
    return seq_saturation_percent


def generate_SUA_plots(adata: sc.AnnData, is_all_cells: bool) -> tuple[str, str]:
    """
    Generate barplot and histogram HTML for spliced, unspliced, and ambiguous RNA counts and splice ratios.

    Parameters
    ----------
    adata
        AnnData object with spliced, unspliced, and ambiguous layers.
    is_all_cells
        True if all cells are included, False for retained cells only.

    Returns
    -------
    Tuple of HTML strings for the barplot and histogram.
    """
    # calculate the sum count for S, U, A matrix separately

    SUA_sum_list = []
    for layer in ["spliced", "unspliced", "ambiguous"]:
        sum_count = int(adata.layers[layer].sum())
        SUA_sum_list.append(sum_count)
    # calculate the S+A/(S+U+A) ratio for each cell(row)
    # get the gene-wise sum for S and U matrix

    S_sum = adata.layers["spliced"].sum(axis=1).A1
    U_sum = adata.layers["unspliced"].sum(axis=1).A1
    A_sum = adata.layers["ambiguous"].sum(axis=1).A1
    # Ensure valid indices
    denominator = S_sum + A_sum + U_sum
    valid_indices = (denominator != 0) & ~np.isnan(denominator)

    # Calculate the ratio only for valid indices
    S_ratio = (S_sum + A_sum)[valid_indices] / denominator[valid_indices]

    # Convert to a Python list
    S_list = S_ratio.tolist()
    # # plot the barplot for S, U, A sum count

    # Create the barplot
    labels = ["Spliced", "Unspliced", "Ambiguous"]
    fig_SUA_bar = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=SUA_sum_list,
                marker={
                    "color": ["#636EFA", "#EF553B", "#00CC96"],
                    "line": {"color": "grey", "width": 1.5},  # Add a border to bars
                },
                text=SUA_sum_list,  # Display numbers on top of bars
                textposition="outside",  # Position the numbers outside the bars
            )
        ]
    )
    title_suffix = get_cell_label(is_all_cells)

    fig_SUA_bar.update_layout(
        title="Bar plot for S, U, A counts" + title_suffix,
        xaxis={"title": "RNA splicing status"},
        yaxis={"title": "Total Counts"},
        # plot_bgcolor='rgba(240,240,240,0.9)',  # Light background
        bargap=0.30,  # Spacing between bars
        margin={"t": 40},
        width=width,
        title_x=0.5,
    )
    apply_uniform_style(fig_SUA_bar)
    # Convert the barplot to HTML
    fig_SUA_bar_html = fig_SUA_bar.to_html(full_html=False, include_plotlyjs="cdn")

    # plot the histogram for S+A/(S+U+A) ratio
    fig_S_ratio = go.Figure(
        data=[
            go.Histogram(
                x=S_list,
            )
        ]
    )

    # Customize the layout
    fig_S_ratio.update_layout(
        title="Histogram of Splicing Ratio" + title_suffix,
        xaxis={"title": "(S+A)/(S+U+A) Ratio"},
        yaxis={"title": "Counts"},
        bargap=0.1,  # Space between bars
        title_x=0.5,
        margin={"t": 40},
    )
    apply_uniform_style(fig_S_ratio)
    # Convert the histogram to HTML
    fig_S_ratio_html = fig_S_ratio.to_html(full_html=False, include_plotlyjs="cdn")

    return fig_SUA_bar_html, fig_S_ratio_html


def umi_dedup(data: pd.DataFrame) -> tuple[go.Figure, float]:
    """
    Generate a scatter plot showing UMI deduplication rates across retained cells.

    Parameters
    ----------
    data
        Feature dump data.

    Returns
    -------
        Tuple of Plotly Figure and mean deduplication rate.
    """
    width = 600
    opacity = 0.5
    # we will use scatter plot to show the reads per CB before and after umi collapse
    before_dedup = np.array(data["mapped_reads"])
    after_dedup = np.array(data["deduplicated_reads"])

    dedup_rate = np.array(data["dedup_rate"])
    mean_dedup_rate = round(np.mean(dedup_rate) * 100, 2)

    # Plot for dedup rate against mapped reads per cell
    fig_dedup = px.scatter(
        x=before_dedup,
        y=after_dedup,
        custom_data=[dedup_rate],
        title="UMI Deduplication rate across Retained Cells",
        labels={"x": "Mapped reads per cell", "y": "Deduplicated UMI per Cell"},
        width=width,
        height=height,
        opacity=opacity,
    )

    # Add y = x reference line
    min_val = min(before_dedup.min(), after_dedup.min())
    max_val = max(before_dedup.max(), after_dedup.max())

    # add mouse hover template
    fig_dedup.update_traces(
        hovertemplate="<br>".join(
            [
                "Dedup Rate: %{customdata:.2%}",
                "Mapped reads: %{x}",
                "Deduplicated UMIs: %{y}",
            ]
        )
    )
    # Add a dashed line with the mean deduplication rate as slope
    fig_dedup.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val * mean_dedup_rate / 100, max_val * mean_dedup_rate / 100],
            mode="lines",
            line={"color": "darkorange", "width": 2, "dash": "dash"},
            name=f"Mean Dedup Rate ({mean_dedup_rate}%)",
            showlegend=False,
        )
    )
    # Add a centered annotation for the Mean Dedup Rate
    fig_dedup.update_layout(
        annotations=[
            {
                "text": f"Mean Dedup Rate: {mean_dedup_rate}%",
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
                "xanchor": "center",
            }
        ]
    )
    return apply_uniform_style(fig_dedup), mean_dedup_rate


def mitochondria_plot(adata: sc.AnnData, is_all_cells: bool) -> go.Figure:
    """
    Generate a violin plot showing mitochondrial content for cells.

    Parameters
    ----------
    adata
        AnnData object for gene expression data.
    is_all_cells
        Whether to include all cells or only retained cells.

    Returns
    -------
    A Plotly Figure visualizing mitochondrial content distribution.
    """
    dot_size = 2
    dot_opacity = 0.3  # Adjust opacity of the dots
    violin_color = "#559364"  # Change color of the violin plot
    scatter_color = "#4F69C2"  # Change color of the scatter plot

    sc.settings.verbosity = 0
    # # we have to filter out cells with less than 1 genes detected, otherwise will will have many cells with high mitochondrial content, which may be misleading of the distribution for valid data.
    if is_all_cells:
        sc.pp.filter_cells(adata, min_genes=20)

    # Identify mitochondrial genes
    # make it case insensitive, to avoid the NaN issue
    adata.var["mt"] = adata.var["gene_symbol"].str.startswith("MT-") | adata.var["gene_symbol"].str.startswith("mt-")
    if adata.var["mt"].sum() > 0:
        # Compute QC metrics
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=[20], log1p=True)
        df = adata.obs[["pct_counts_mt"]].copy()
    else:
        logger.warning("No mitochondrial genes found.")
        # Create an empty DataFrame to generate a blank violin plot
        df = pd.DataFrame({"pct_counts_mt": [], "Sample": []})

    # Convert to Pandas DataFrame

    df["sample"] = "Cells"  # Ensure a single category for violin plot
    df["sample"] = df["sample"].astype(str)  # Ensure categorical behavior

    title_suffix = get_cell_label(is_all_cells)
    # Create the violin plot
    fig_mito = px.violin(
        df,
        x="sample",
        y="pct_counts_mt",
        box=True,  # Show box plot inside violin
        points=False,  # Don't show points here (we'll add manually)
        title="Mitochondrial Content(%)" + title_suffix,
        labels={"pct_counts_mt": "Percentage of Mitochondrial Counts"},
        width=width,
        height=height,
        color_discrete_sequence=[violin_color],
    )

    # Create the scatter (dot) plot manually
    scatter = px.strip(
        df,
        x="sample",
        y="pct_counts_mt",
    )

    # Add scatter points to the violin plot
    for trace in scatter.data:
        trace.marker.size = dot_size
        trace.marker.opacity = dot_opacity
        trace.marker.color = scatter_color
        fig_mito.add_trace(trace)

    return apply_uniform_style(fig_mito)


def umap_tsne_plot(
    adata: sc.AnnData,
) -> tuple[go.Figure, go.Figure, str]:
    """
    Perform dimensionality reduction and clustering, and generate UMAP and t-SNE plots.

    Parameters
    ----------
    adata
        AnnData object for gene expression mtx.

    Returns
    -------
    tuple
        - A Plotly Figure for the UMAP plot.
        - A Plotly Figure for the t-SNE plot.
        - A string containing a code snippet for preprocessing and visualization.
    """
    sc.settings.set_figure_params(dpi=200, facecolor="white")

    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)

    # feature selection
    n_valid = adata.X.shape[1]
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, n_valid))
    # dimensionality Reduction
    sc.tl.pca(adata)
    # nearest neighbor graph constuction and visualization
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # clustering
    # Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    n_cells = adata.n_obs
    perplexity = min(30, max(2, (n_cells - 1) // 3))
    if perplexity < 30:
        logger.warning(
            f"🛠️ Perplexity is set to {perplexity} (default is 30) for t-SNE due to a low number of valid cells."
        )
    try:
        sc.tl.tsne(adata, perplexity=perplexity)
    except TypeError as e:
        if "PCA initialization is currently not supported with the sparse input matrix" in str(e):
            logger.warning("🛠️ t-SNE failed on sparse matrix; converting to dense.")
            adata.X = adata.X.toarray()
            sc.tl.tsne(adata, perplexity=perplexity)
        else:
            raise

    # Create a Plotly-based UMAP scatter plot with Leiden clusters
    umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    umap_df["leiden"] = adata.obs["leiden"].values

    # UMAP in plotly
    opacity = 0.7
    # modify dot size
    dot_size = 3
    fig_umap = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="leiden",
        title="UMAP with Leiden Clusters (Retained Cells Only)",
        width=width,
        height=height,
        opacity=opacity,
    ).update_traces(marker={"size": dot_size})  # Set the dot size

    # Center title and reduce margin
    fig_umap.update_layout(title_x=0.5, margin={"t": 30, "l": 10, "r": 10, "b": 20})
    # t-SNE plot in plotly
    tsne_df = pd.DataFrame(adata.obsm["X_tsne"], columns=["TSNE1", "TSNE2"])
    tsne_df["leiden"] = adata.obs["leiden"].values

    fig_tsne = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="leiden",
        title="t-SNE with Leiden Clusters (Retained Cells Only)",
        width=480,
        height=360,
        opacity=0.7,
    ).update_traces(marker={"size": 3})
    # set up the code block
    code_text = """
        # Pre-processing for UMAP and t-SNE

        sc.settings.set_figure_params(dpi=200, facecolor="white")
        # Normalizing to median total counts
        sc.pp.normalize_total(adata)
        # Logarithmize the data
        sc.pp.log1p(adata)

        # feature selection
        n_valid = adata.X.shape[1]
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, n_valid))
        # dimensionality Reduction
        sc.tl.pca(adata)
        # nearest neighbor graph constuction and visualization
        sc.pp.neighbors(adata)
        # UMAP
        sc.tl.umap(adata)

        # clustering
        # Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

        n_cells = adata.n_obs
        perplexity = min(30, max(2, (n_cells - 1) // 3))

        # t-SNE
        sc.tl.tsne(adata, perplexity=perplexity)

        # Create a Plotly-based UMAP scatter plot with Leiden clusters
        umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        umap_df["leiden"] = adata.obs["leiden"].values

        # UMAP in plotly
        opacity = 0.7
        # modify dot size
        dot_size = 3
        fig_umap = px.scatter(
            umap_df,
            x="UMAP1",
            y="UMAP2",
            color="leiden",
            title="UMAP with Leiden Clusters (Retained Cells Only)",
            width=width,
            height=height,
            opacity=opacity,
        ).update_traces(marker={"size": dot_size})  # Set the dot size

        # Center title and reduce margin
        fig_umap.update_layout(title_x=0.5, margin={"t": 30, "l": 10, "r": 10, "b": 20})

        # t-SNE plot in plotly
        tsne_df = pd.DataFrame(adata.obsm["X_tsne"], columns=["TSNE1", "TSNE2"])
        tsne_df["leiden"] = adata.obs["leiden"].values

        fig_tsne = px.scatter(
            tsne_df,
            x="TSNE1",
            y="TSNE2",
            color="leiden",
            title="t-SNE with Leiden Clusters (Retained Cells Only)",
            width=480,
            height=360,
            opacity=0.7,
        ).update_traces(marker={"size": 3})
        """

    fig_tsne.update_layout(title_x=0.5, margin={"t": 30, "l": 10, "r": 10, "b": 20})
    return apply_uniform_style(fig_umap), apply_uniform_style(fig_tsne), code_text


def show_quant_log_table(quant_json_data: dict, permit_list_json_data: dict | None) -> tuple[str, str]:
    """
    Generate an HTML table from quant log and permit list JSON data.

    Parameters
    ----------
    quant_json_data
        log from the 'quant' step of alvin-fry
    permit_list_json_data
        log from the 'generate-permit-list' step of alvin-fry

    Returns
    -------
    tuple
        Two HTML strings: one for the quant info table and one for the permit list table.
    """
    collapse_threshold = 30
    quant_table_content = ""
    permit_list_table_content = ""

    key_list_lv1 = list(quant_json_data.keys())

    for key_lv1 in key_list_lv1:
        table_row = f"<tr><td>{key_lv1}</td>"

        if key_lv1 == "empty_resolved_cell_numbers" and len(quant_json_data[key_lv1]) > collapse_threshold:
            # Format the list of numbers as a comma-separated string
            formatted_content = ", ".join(map(str, quant_json_data[key_lv1]))

            # Generate Bootstrap accordion for the content
            accordion_content = f"""
            <div class="accordion" id="accordion_{key_lv1}">
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading_{key_lv1}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse_{key_lv1}" aria-expanded="false" aria-controls="collapse_{key_lv1}">
                            Expand Content
                        </button>
                    </h2>
                    <div id="collapse_{key_lv1}" class="accordion-collapse collapse" aria-labelledby="heading_{key_lv1}" data-bs-parent="#accordion_{key_lv1}">
                        <div class="accordion-body">
                            {formatted_content}
                        </div>
                    </div>
                </div>
            </div>
            """
            quant_table_content += f"{table_row}<td>{accordion_content}</td></tr>"
        else:
            # Directly display the content for other cases
            quant_table_content += f"{table_row}<td>{str(quant_json_data[key_lv1])}</td></tr>"

    # Create Permit List Table
    if permit_list_json_data:
        permit_list_table_content = ""
        for key, value in permit_list_json_data.items():
            permit_list_table_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
    else:
        permit_list_table_content = "<tr><td colspan='3'>No permit list data available.</td></tr>"

    return (quant_table_content, permit_list_table_content)


def generate_summary_table(
    raw_featuredump_data: pd.DataFrame,
    valid_bcs: list[str],
    total_detected_genes: int,
    median_genes_per_cell: int,
    mapping_rate: float | None,
    seq_saturation_value: float,
) -> str:
    """
    Generate a summary table with total corrected reads, median UMI, and total number of cells, etc.

    Parameters
    ----------
    raw_featuredump_data
        Feature dump data including read counts and barcodes.
    valid_bcs
        Barcodes retained after filtering.
    total_detected_genes
        Total number of genes detected in retained cells.
    median_genes_per_cell
        Median number of genes detected per retained cell.
    mapping_rate
        Fraction of reads mapped to the reference transcriptome, or None.
    seq_saturation_value
        Percentage of sequencing saturation.

    Returns
    -------
    tuple
        - An HTML string representing the summary table.
        - summary metrics.
    """
    total_cells = len(raw_featuredump_data["corrected_reads"])
    # filter with valid barcodes
    data = raw_featuredump_data[raw_featuredump_data["barcodes"].isin(valid_bcs)]
    mean_reads_per_cell = int(np.nan_to_num(data["corrected_reads"].mean(), nan=0))
    median_umi_per_cell = int(np.nan_to_num(data["deduplicated_reads"].median(), nan=0))

    summary = {
        "Number of retained cells": f"{len(valid_bcs):,}",
        "Number of all processed cells": f"{total_cells:,}",
        "Mean reads per retained cell": f"{mean_reads_per_cell:,}",
        "Median UMI per retained cell": f"{median_umi_per_cell:,}",
        "Median genes per retained cell": f"{median_genes_per_cell:,}",
        "Total genes detected for retained cells": f"{total_detected_genes:,}",
        "Mapping rate": f"{mapping_rate}%" if mapping_rate else "N/A",
        "Sequencing saturation": f"{seq_saturation_value}%",
    }

    # Convert summary dict to a list of (key, value) pairs
    summary_items = list(summary.items())

    summary_table_rows = ""
    # Process pairs in steps of 2
    for i in range(0, len(summary_items), 2):
        # Left key-value
        left_k, left_v = summary_items[i]

        # Right key-value, if it exists
        if i + 1 < len(summary_items):
            right_k, right_v = summary_items[i + 1]
        else:
            # If there's an odd total, leave these blank
            right_k, right_v = "", ""

        # Build a single row (4 columns)
        summary_table_rows += (
            f"<tr>"
            f"<td class='key-column'>{left_k}</td><td class='value-cell'>{left_v}</td>"
            f"<td class='key-column'>{right_k}</td><td class='value-cell'>{right_v}</td>"
            f"</tr>"
        )

    # Wrap rows in a Bootstrap .table
    summary_html = f"""
    <div class="table-responsive">
      <table class="table">
        <tbody>
          {summary_table_rows}
        </tbody>
      </table>
    </div>
    """

    return summary_html, summary
