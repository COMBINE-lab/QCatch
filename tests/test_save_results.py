from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from qcatch.input_processing import save_results


def test_save_results_preserves_x_and_named_layers(tmp_path):
    counts = np.array([[1, 2], [3, 4]], dtype=np.float32)
    adata = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=pd.DataFrame({"barcodes": ["b", "a"]}, index=["b", "a"]),
        var=pd.DataFrame(index=pd.Index(["z", "y"], name="gene_ids")),
    )
    adata.layers["unspliced"] = sparse.csr_matrix(counts * 2)
    adata.layers["spliced"] = sparse.csr_matrix(counts * 3)
    args = SimpleNamespace(
        input=SimpleNamespace(mtx_data=adata, is_h5ad=True, dir=tmp_path / "input"),
        output=tmp_path,
        valid_cell_list=True,
        save_filtered_h5ad=True,
    )
    save_results(args, "test", None, ["a"])

    full = ad.read_h5ad(tmp_path / "quants.h5ad")
    filtered = ad.read_h5ad(tmp_path / "filtered_quants.h5ad")
    expected = counts[::-1, ::-1]
    np.testing.assert_array_equal(full.X.toarray(), expected)
    np.testing.assert_array_equal(filtered.X.toarray(), expected[:1])
    for name, multiplier in [("spliced", 3), ("unspliced", 2)]:
        np.testing.assert_array_equal(full.layers[name].toarray(), expected * multiplier)
        np.testing.assert_array_equal(filtered.layers[name].toarray(), expected[:1] * multiplier)
    assert full.obs_names.tolist() == ["a", "b"]
    assert full.obs["is_retained_cells"].tolist() == [True, False]
    assert filtered.obs_names.tolist() == ["a"]
