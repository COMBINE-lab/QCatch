# QCatch
Quality Control downstream of alevin-fry / simpleaf
=======

## How to install

## How to use

## Disclaimer


## Command-Line Arguments

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--input`  | `-i` | `str` (Required) | Path to the input directory containing the quant output files. |
| `--output` | `-o` | `str` (Optional) | Path to the output directory. **Default**: Same directory as input. |
| `--chemistry` | `-c` | `str` (Optional) | Specifies the chemistry used in the experiment, determining the range for the `empty_drops` step. **Options**: `'10xv2'`, `'10xv3'`, `'10xv4'`, `'3p_lt'`. **Default**: Will use the range for `'10xv2'` and `'10xv3'`. |
| `--gene_id2name_dir` | `-g` | `str` (Optional) | Directory containing the `gene_id2name` file for converting Ensembl gene IDs to gene names. The file must be a CSV with two columns: `gene_id` and `gene_name`. If not provided, mitochondria plots requiring gene names will not be displayed. **Default**: `None`. |
| `--out_prefix` | `-p` | `str` (Optional) | Prefix for the output files. **Default**: `None`. |
