import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pytest

import qcatch
from qcatch.pp import elaborate_example


def test_package_has_version():
    assert qcatch.__version__ is not None


@pytest.mark.parametrize(
    "transform,layer_key,max_items,expected_len,expected_substring",
    [
        # Test default parameters
        (lambda vals: f"mean={vals.mean():.2f}", None, 100, 1, "mean="),
        # Test with layer_key
        (lambda vals: f"mean={vals.mean():.2f}", "scaled", 100, 1, "mean=0."),
        # Test with max_items limit (won't affect single item)
        (lambda vals: f"max={vals.max():.2f}", None, 1, 1, "max=6.70"),
    ],
)
def test_elaborate_example_adata_only_simple(
    adata,  # this tests uses the adata object from the fixture in the conftest.py
    transform,
    layer_key,
    max_items,
    expected_len,
    expected_substring,
):
    result = elaborate_example(items=[adata], transform=transform, layer_key=layer_key, max_items=max_items)

    assert len(result) == expected_len
    assert expected_substring in result[0]
