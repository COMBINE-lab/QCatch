import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from pyroe import load_fry
import logging
import scanpy as sc

logger = logging.getLogger(__name__)

def apply_uniform_style(fig):
    """Apply a consistent style with a solid background matching the container."""
    fig.update_layout(
        title_x=0.5,
        margin=dict(t =40, l=30, r=30, b=30),
        plot_bgcolor='rgba(0,0,0,0)',  # Fully transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=14, color="#2c3e50"),
        title_font=dict(size=16, color="#34495e"),
        xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='#34495e', tickfont=dict(size=14, color='#34495e')),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='#34495e', tickfont=dict(size=14, color='#34495e')),
    )
    return fig

def get_cell_label(is_all_cells):
    return " (All Cells)" if is_all_cells else " (Retained Cells Only)"
# set up width and height for the plots in horizontal 
width = 480
height = 360
opacity = 0.5

def generate_knee_plots(data):
    width=  400
    height= 400
    
    # 1.1 Knee Plot 1: Rank vs UMI Counts
    fig_knee_1 = px.scatter(
        data,
        x="rank",
        y="deduplicated_reads",
        log_x=True,
        log_y=True,
        title="UMI Counts vs Cell Rank (All Cells)",
        labels={"rank": "Cell Rank", "deduplicated_reads": "UMI Counts"},
        width=width,
        height=height,
        opacity=opacity
    )
    
    
    # 1.2 Knee Plot 2: Rank vs Genes Detected
    fig_knee_2 = px.scatter(
    data,
    x="rank",
    y="num_expressed",
    log_x=True,
    log_y=True,
    title="Genes Detected against Cell Rank (All Cells)",
    labels={"rank": "Cell Rank", "num_expressed": "Number of Detected Genes"},
    width=width,
    height=height,
    opacity=opacity
    )
    return apply_uniform_style(fig_knee_1), apply_uniform_style(fig_knee_2)

def generate_gene_histogram(data, is_all_cells):
    # set up width and height for the plots in horizontal&1 fig per tab
    
    gene_detected = data["num_expressed"].tolist()
    title_suffix = get_cell_label(is_all_cells)
    
    # Create a histogram plot
    fig_hist_genes = go.Figure(
        data=[go.Histogram(
            x=gene_detected,
            name="Number of detected genes",
            opacity=0.8,
        )],
        layout=go.Layout(
            title="Histogram of detected genes per cell"+title_suffix,
            xaxis=dict(
                title="Number of detected genes",
                autorange=True
            ),
            yaxis=dict(title="Count", type='log', dtick=1),
            width=width,
            height=height,
            bargap=0.05
        )
    )
    return apply_uniform_style(fig_hist_genes)

def generate_seq_saturation(data):
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

    correct_reads_array = unfiltered_correct_reads_array[idx_start:]
    seq_saturation_array = unfiltered_seq_saturation_array[idx_start:]

    # Calculate average sequencing saturation value for all filtered cells
    seq_saturation_value = sum(seq_saturation_array) / len(seq_saturation_array) if seq_saturation_array else 0
    seq_saturation_percent = round(seq_saturation_value * 100, 2)

    # Down-sampling for Sequencing Saturation
    accumu_read_per_cell_array = [0]
    accumu_seq_saturation_array = [0]
    bins = 10
    num_of_cells = len(correct_reads_array)

    for i in range(1, num_of_cells // bins):
        downsample_correct_reads_array = correct_reads_array[:bins * i]
        downsample_seq_saturation_array = seq_saturation_array[:bins * i]

        down_mean_reads_per_cell = sum(downsample_correct_reads_array) / len(downsample_correct_reads_array)
        down_seq_saturation_value = sum(downsample_seq_saturation_array) / len(downsample_seq_saturation_array)

        accumu_read_per_cell_array.append(down_mean_reads_per_cell)
        accumu_seq_saturation_array.append(down_seq_saturation_value)
        
    # for the accumulated arrays, only use 15 dots for plotting
    num_dots = 20
    # avoid the case where the number of points is less than num_dots
    step = max(1, len(accumu_read_per_cell_array) // num_dots)
    accumu_read_per_cell_array = accumu_read_per_cell_array[::step]
    accumu_seq_saturation_array = accumu_seq_saturation_array[::step]
    # Plot for Sequencing Saturation using go.Scatter with smooth line
    fig_seq_saturation = go.Figure(
        data=go.Scatter(
            x=accumu_read_per_cell_array,
            y=accumu_seq_saturation_array,
            mode='lines',
            line=dict(color='#636EFA', width=3)
        )
    )
    fig_seq_saturation.update_layout(
        title="Sequencing Saturation Plot (Retained Cells Only)",
        xaxis_title="Mean Reads per Cell",
        yaxis=dict(title="Sequencing Saturation", range=[0, 1]),
        width=width,
        height=height,
        margin=dict(t=40)
    )
    fig_seq_saturation.add_shape(
        type="line",
        x0=min(accumu_read_per_cell_array),
        x1=max(accumu_read_per_cell_array),
        y0=0.9,
        y1=0.9,
        line=dict(color="lightgrey", width=2, dash="dash"),
        layer="below"
    )

    return apply_uniform_style(fig_seq_saturation), seq_saturation_percent

def generate_SUA_plots(adata, is_all_cells):
    # calculate the sum count for S, U, A matrix separately

    SUA_sum_list = []
    for layer in ['spliced', 'unspliced', 'ambiguous']:
        sum_count = int(adata.layers[layer].sum())
        SUA_sum_list.append(sum_count)
    # calculate the S/(S+U) ratio for each cell(row)
    # get the gene-wise sum for S and U matrix

    S_sum = adata.layers['spliced'].sum(axis=1).tolist()
    U_sum = adata.layers['unspliced'].sum(axis=1).tolist()

    # Ensure valid indices
    denominator = np.array(S_sum) + np.array(U_sum)
    valid_indices = (denominator != 0) & ~np.isnan(denominator)

    # Calculate the ratio only for valid indices
    S_ratio = np.array(S_sum)[valid_indices] / denominator[valid_indices]

    # Convert to a Python list
    S_list = S_ratio.tolist()
    # # plot the barplot for S, U, A sum count

    # Create the barplot
    labels = ['Spliced', 'Unspliced', 'Ambiguous']
    fig_SUA_bar = go.Figure(
        data=[go.Bar(
            x=labels,
            y=SUA_sum_list,
            marker=dict(
                color=['#636EFA', '#EF553B', '#00CC96'] ,
                line=dict(color='grey', width=1.5)  # Add a border to bars
            ),
            text=SUA_sum_list,  # Display numbers on top of bars
            textposition='outside'  # Position the numbers outside the bars
        )]
    )
    title_suffix = get_cell_label(is_all_cells)
    
    fig_SUA_bar.update_layout(
        title="Bar plot for S, U, A counts"+title_suffix,
        xaxis=dict(title="RNA splicing status"),
        yaxis=dict(title="Total Counts"),
        # plot_bgcolor='rgba(240,240,240,0.9)',  # Light background
        bargap=0.30,  # Spacing between bars
        margin=dict(t=40),
        width=width,
        title_x=0.5, 

    )
    apply_uniform_style(fig_SUA_bar)
    # Convert the barplot to HTML
    fig_SUA_bar_html = fig_SUA_bar.to_html(full_html=False, include_plotlyjs="cdn")

    # plot the histogram for S/(S+U) ratio
    fig_S_ratio = go.Figure(
        data=[go.Histogram(
            x=S_list,
        )]
    )

    # Customize the layout
    fig_S_ratio.update_layout(
        title="Histogram of S/(S+U) Ratio"+title_suffix,
        xaxis=dict(title="S/(S+U) Ratio"),
        yaxis=dict(title="Count"),
        bargap=0.1,  # Space between bars
        title_x=0.5,
        margin=dict(t=40)
    )
    apply_uniform_style(fig_S_ratio)
    # Convert the histogram to HTML
    fig_S_ratio_html = fig_S_ratio.to_html(full_html=False, include_plotlyjs="cdn")

    return fig_SUA_bar_html, fig_S_ratio_html

def barcode_collapse(data):
    opacity = 0.5
    # we will use scatter plot to show the reads per CB before and after collapsing
    
    before_clap = np.array(data["mapped_reads"])
    after_clap = np.array(data["corrected_reads"])

    # Compute the gain_rate safely (avoid division by zero)
    gain_rate = (after_clap - before_clap) / before_clap
    mean_gain_rate = round(np.mean(gain_rate)*100, 4)
    # Plot for Barcode Collapse
    fig_barcode_collapse = px.scatter(
        x=before_clap,
        y=after_clap,
        title="Barcode Collapsing (Retained Cells Only)",
        labels={"x": "Reads per Cell before collapsing", "y": "Reads per Cell after collapsing"},
        width=width,
        height=height,
        opacity=opacity
    )
    # Add y = x reference line
    min_val = min(before_clap.min(), after_clap.min())  # Get min value from both axes
    max_val = max(before_clap.max(), after_clap.max())  # Get max value from both axes
    fig_barcode_collapse.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#ed8888", width=2),
            showlegend=False,
            # name="y = x Line"
        )
    )
    
    
    return apply_uniform_style(fig_barcode_collapse), mean_gain_rate

def barcode_frequency_plots(data):
    # scatter plot for reads per CB v.s. UMI counts in this CB
    # width = 550
    height = 380
    opacity = 0.5

    fig_bc_freq_UMI = px.scatter(
        data, 
        x="corrected_reads", 
        y="deduplicated_reads", 
        # log_y=True,
        title="Barcode frequency vs UMI Counts(All Cells)", 
        labels={"corrected_reads": "Barcode frequency", "deduplicated_reads": "UMI Counts"},
        width=width,
        height=height,
        opacity=opacity)
    
    # scatter plot for barcode frequency v.s. number of genes detected
    fig_bc_freq_gene= px.scatter(
        data, 
        x="corrected_reads", 
        y="num_expressed", 
        # log_y=True, 
        title="Barcode frequency vs Number of Genes Detected(All Cells)", 
        labels={"corrected_reads": "Barcode frequency", "num_expressed": "Number of Genes Detected"},
        width=width,
        height=height,
        opacity=opacity)
    
    # UMI Counts vs Number of Detected Genes
    fig_gene_UMI = px.scatter(
        data,
        x="deduplicated_reads",
        y="num_expressed",
        # log_y=True,
        title="UMI Counts vs Number of Detected Genes(All Cells)",
        labels={"deduplicated_reads": "UMI Counts", "num_expressed": "Number of Detected Genes"},
        width=width,
        height=height,
        opacity=opacity
        )
    
    return apply_uniform_style(fig_bc_freq_UMI), apply_uniform_style(fig_bc_freq_gene), apply_uniform_style(fig_gene_UMI)
    
def mitochondria_plot(adata,is_all_cells):
    """Generates a properly aligned Plotly violin plot for mitochondrial content."""
    dot_size = 2
    dot_opacity = 0.3  # Adjust opacity of the dots
    violin_color = '#559364'  # Change color of the violin plot
    scatter_color = '#4F69C2'  # Change color of the scatter plot

    sc.settings.verbosity = 0
    # # we have to filter out cells with less than 1 genes detected, otherwise will will have many cells with high mitochondrial content, which may be misleading of the distribution for valid data.
    if is_all_cells:
        sc.pp.filter_cells(adata, min_genes=10)
    
    # Identify mitochondrial genes
    # make it case insensitive, to avoid the NaN issue
    adata.var["mt"] = adata.var['gene_symbol'].str.startswith("MT-") | adata.var['gene_symbol'].str.startswith("mt-")
    if adata.var["mt"].sum() > 0:
        # Compute QC metrics
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=[20], log1p=True)   
        df = adata.obs[['pct_counts_mt']].copy()
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
        title="Mitochondrial Content(%)"+title_suffix,
        labels={"pct_counts_mt": "Percentage of Mitochondrial Counts"},
        width=width,
        height=height,
        color_discrete_sequence=[violin_color]  
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

def umap_tsne_plot(adata):
    sc.settings.set_figure_params(dpi=200, facecolor="white")
    
    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    
    # feature selection
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    # dimensionality Reduction
    sc.tl.pca(adata)
    # nearest neighbor graph constuction and visualization
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    # clustering
    # Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    sc.tl.tsne(adata)

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
        opacity=opacity
    ).update_traces(marker=dict(size=dot_size))  # Set the dot size

    # Center title and reduce margin
    fig_umap.update_layout(
        title_x=0.5,
        margin=dict(t=30, l=10, r=10, b=20)
    )
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
        opacity=0.7
    ).update_traces(marker=dict(size=3))

    fig_tsne.update_layout(
        title_x=0.5,
        margin=dict(t=30, l=10, r=10, b=20)
    )
    return apply_uniform_style(fig_umap), apply_uniform_style(fig_tsne)

def show_quant_log_table(quant_json_data, permit_list_json_data):
    """
    Generate a Bootstrap-styled HTML table from the JSON data.
    """
    
    collapse_threshold = 30
    quant_table_content = ""
    permit_list_table_content = ""

    key_list_lv1 = list(quant_json_data.keys())
    current_index = 1

    for key_lv1 in key_list_lv1:
        table_row = f"<tr><th scope='row'>{current_index}</th><td>{key_lv1}</td>"

        if key_lv1 == "empty_resolved_cell_numbers" and len(quant_json_data[key_lv1]) > collapse_threshold:
            # Format the list of numbers as a comma-separated string
            formatted_content = ", ".join(map(str, quant_json_data[key_lv1]))

            # Generate Bootstrap accordion for the content
            accordion_content = f'''
            <div class="accordion" id="accordion{current_index}">
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading{current_index}">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{current_index}" aria-expanded="false" aria-controls="collapse{current_index}">
                            Expand Content
                        </button>
                    </h2>
                    <div id="collapse{current_index}" class="accordion-collapse collapse" aria-labelledby="heading{current_index}" data-bs-parent="#accordion{current_index}">
                        <div class="accordion-body">
                            {formatted_content}
                        </div>
                    </div>
                </div>
            </div>
            '''
            quant_table_content += f"{table_row}<td>{accordion_content}</td></tr>"
        else:
            # Directly display the content for other cases
            quant_table_content += f"{table_row}<td>{quant_json_data[key_lv1]}</td></tr>"

        current_index += 1

    # Create Permit List Table 
    if permit_list_json_data:
        permit_list_table_content = ""
        for key, value in permit_list_json_data.items():
            permit_list_table_content += f"<tr><th scope='row'>{current_index}</th><td>{key}</td><td>{value}</td></tr>"
            current_index += 1
    else: 
        permit_list_table_content = "<tr><td colspan='3'>No permit list data available.</td></tr>"

    return quant_table_content, permit_list_table_content


def generate_summary_table(raw_data, valid_bcs, total_detected_genes):
    """
    Generate a summary table with total corrected reads, total UMI, and total number of cells, etc.
    in a 4-column (two key-value pairs per row) Bootstrap table.
    """
    total_cells = len(raw_data['corrected_reads'])
    # filter with valid barcodes
    data = raw_data[raw_data['barcodes'].isin(valid_bcs)]
    mean_reads_per_cell = int(np.nan_to_num(data['corrected_reads'].mean(), nan=0))
    median_umi_per_cell = int(np.nan_to_num(data['deduplicated_reads'].median(), nan=0))
    median_genes_per_cell = int(np.nan_to_num(data['num_expressed'].median(), nan=0))

    # total_correct_reads = sum(data['corrected_reads'])
    # total_UMI = sum(data['deduplicated_reads'])
    summary = {
        "Number of retained cells": f"{len(valid_bcs):,}",
        "Number of all processed cells": f"{total_cells:,}",
        "Mean reads per retained cell": f"{mean_reads_per_cell:,}",
        "Median UMI per retained cell": f"{median_umi_per_cell:,}",
        "Median genes per retained cell": f"{median_genes_per_cell:,}",
        "Total genes detected for retained cells": f"{total_detected_genes:,}",
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

    return summary_html
