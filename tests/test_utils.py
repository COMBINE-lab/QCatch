import pandas as pd
import plotly.graph_objs as go

from qcatch.plots_tables import (
    generate_gene_histogram,
    generate_seq_saturation,
    get_cell_label,
    umi_dedup,
)


def test_get_cell_label():
    assert "All Cells" in get_cell_label(True)
    assert "Retained Cells Only" in get_cell_label(False)


def test_generate_seq_saturation_basic():
    df = pd.DataFrame(
        {
            "corrected_reads": [10, 50, 200, 500],
            "dedup_rate": [0.8, 0.7, 0.6, 0.5],
        }
    )
    val = generate_seq_saturation(df)
    assert isinstance(val, float)
    assert 0.0 <= val <= 100.0


def _mini_fd():
    return pd.DataFrame(
        {
            "barcodes": ["A", "B", "C", "D"],
            "rank": [1, 2, 3, 4],
            "deduplicated_reads": [100, 50, 20, 10],
            "num_genes_expressed": [500, 300, 100, 50],
            "corrected_reads": [120, 60, 25, 12],
            "mapped_reads": [150, 70, 30, 15],
            "dedup_rate": [0.7, 0.6, 0.5, 0.4],
        }
    )


def test_generate_gene_histogram_smoke():
    fig = generate_gene_histogram(_mini_fd(), True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_umi_dedup_smoke():
    fig, mean_rate = umi_dedup(_mini_fd())
    assert isinstance(fig, go.Figure)
    assert isinstance(mean_rate, float)
    assert 0 <= mean_rate <= 100
