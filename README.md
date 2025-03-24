# QCatch
Quality Control downstream of alevin-fry / simpleaf.

**QCatch** is a Python package designed to streamline quality control for single-cell sequencing data quantified by [alevin-fry](https://github.com/COMBINE-lab/alevin-fry) or [simpleaf](https://github.com/COMBINE-lab/simpleaf). It provides a comprehensive web-based quality control report, enabling researchers to:

- Summarize key **quality metrics** for single-cell sequencing datasets.
- Perform **cell calling** to identify high-quality cells.
- Generate interactive **visualizations** to support downstream analysis and interpretation.

QCatch is built to simplify the quality control process, making it easier for researchers to assess data quality and make informed decisions for further analysis.

## Installation

<!-- ### Bioconda
You can install using [Conda](http://anaconda.org/)
from [Bioconda](https://bioconda.github.io/).

```bash
# conda install QCatch (TODO)
```

### PyPl
`QCatch` is also availabel in PyPl.
```bash
# pip install QCatch (TODO)
``` -->

## Usage
Specify the path to the folder containing quantification results generated by `alevin-fry` or `simpleaf`. QCatch will scan the folder, evaluate the data quality, and generate an HTML report that can be viewed directly in your browser.

## Command-Line Arguments

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--input`  | `-i` | `str` (Required) | Path to the input directory containing the quant output files. |
| `--output` | `-o` | `str` | Path to the output directory. **Default**: Same directory as input. |
| `--chemistry` | `-c` | `str` | Specifies the chemistry used in the experiment, determining the range for the `empty_drops` step. **Options**: `'10xv2'`, `'10xv3'`, `'10xv4'`, `'3p_lt'`. **Default**: Will use the range for `'10xv2'` and `'10xv3'`. |
| `--n_partitions` | `-n` | `int` (Optional) | Number of partitions (max number of barcodes to consider for ambient estimation). Skip this if `--chemistry` is specified. |
| `--gene_id2name_dir` | `-g` | `str` (Optional) | Directory containing the `gene_id2name` file for converting Ensembl gene IDs to gene names. The file must be a CSV with two columns: `gene_id` and `gene_name`. If not provided, mitochondria plots requiring gene names will not be displayed. **Default**: `None`. |
| `--verbose` | `-v` | `flag` (Optional) | Enable verbose logging with debug-level messages. |
| `--overwrite_h5ad` | `-o` | `flag` (Optional) | If set, overwrites the original .h5ad file with updated cell filtering results. |

