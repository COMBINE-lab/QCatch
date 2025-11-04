import os
import subprocess
from pathlib import Path


def test_simpleaf_wrong_path(tmp_path):
    input_path = Path("tests/data/test_data/simpleaf_wrong_path/quants.h5ad")
    output_path = tmp_path / "test_output_cookie"
    main_py = Path(__file__).resolve().parent.parent / "src" / "qcatch" / "main.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

    cmd = [
        "python",
        str(main_py),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--chemistry",
        "10X_3p_v3",
        "--verbose",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode != 0


def test_simpleaf_with_map_runs(tmp_path):
    input_path = Path("tests/data/test_data/simpleaf_with_map/quants.h5ad")
    output_path = tmp_path / "test_output_cookie"
    main_py = Path(__file__).resolve().parent.parent / "src" / "qcatch" / "main.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

    cmd = [
        "python",
        str(main_py),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--skip_umap_tsne",
        "--verbose",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    assert output_path.exists()


def test_mtx_input_1k_pbmc_v3(tmp_path):
    input_path = Path("tests/data/test_data/1k_pbmc_v3")
    output_path = tmp_path / "1k_pbmc_v3_out"
    main_py = Path(__file__).resolve().parent.parent / "src" / "qcatch" / "main.py"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

    cmd = [
        "python",
        str(main_py),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--chemistry",
        "10X_3p_v3",
        "--verbose",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    assert output_path.exists()


# def test_simpleaf_with_cb_list(tmp_path):
#     input_path = Path("tests/data/test_data/simpleaf_rerun_1k_pbmc/quants.h5ad")
#     output_path = tmp_path / "test_output_cookie" / "simpleaf_latest_1k_with_cb_list"
#     valid_cb_list = Path("tests/data/test_data/simpleaf_rerun_1k_pbmc/total_retained_cells.tsv")
#     main_py = Path(__file__).resolve().parent.parent / "src" / "qcatch" / "main.py"

#     env = os.environ.copy()
#     env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

#     cmd = [
#         "python",
#         str(main_py),
#         "--input",
#         str(input_path),
#         "--output",
#         str(output_path),
#         "--chemistry",
#         "10X_3p_v3",
#         "--valid_cell_list",
#         str(valid_cb_list),
#         "--skip_umap_tsne",
#         "--export_summary_table",
#         "--verbose",
#     ]

#     result = subprocess.run(cmd, capture_output=True, text=True, env=env)
#     print(result.stdout)
#     print(result.stderr)
#     assert result.returncode == 0
#     assert output_path.exists()


# def test_simpleaf_overwrite_save(tmp_path):
#     input_path = Path("tests/data/test_data/simpleaf_rerun_1k_pbmc_overwrite/quants.h5ad")
#     valid_cb_list = Path("tests/data/test_data/simpleaf_rerun_1k_pbmc/total_retained_cells.tsv")
#     main_py = Path(__file__).resolve().parent.parent / "src" / "qcatch" / "main.py"

#     env = os.environ.copy()
#     env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

#     cmd = [
#         "python",
#         str(main_py),
#         "--input",
#         str(input_path),
#         "--chemistry",
#         "10X_3p_v3",
#         "--valid_cell_list",
#         str(valid_cb_list),
#         "--skip_umap_tsne",
#         "--export_summary_table",
#         "--verbose",
#     ]

#     result = subprocess.run(cmd, capture_output=True, text=True, env=env)
#     print(result.stdout)
#     print(result.stderr)
#     assert result.returncode == 0
