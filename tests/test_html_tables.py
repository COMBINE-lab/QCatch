import pandas as pd

from qcatch.plots_tables import generate_summary_table


def test_generate_summary_table_dict_and_html():
    df = pd.DataFrame(
        {
            "barcodes": ["A", "B", "C"],
            "corrected_reads": [100, 200, 300],
            "deduplicated_reads": [80, 150, 250],
        }
    )
    html, summary = generate_summary_table(
        raw_featuredump_data=df,
        valid_bcs=["A", "C"],
        total_detected_genes=12345,
        median_genes_per_cell=1500,
        mapping_rate=90.5,
        seq_saturation_value=75.0,
    )
    assert "Number of retained cells" in html
    assert summary["Mapping rate"] == "90.5%"
