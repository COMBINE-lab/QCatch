# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html
## [0.2.9] 2026-2
### Added
- `--visualize_doublets` flag to generate toggle-able UMAP/t-SNE plots showing singlets and doublets with shared coordinate space

### Changed
- Mitochondrial plot gene threshold reduced from 20 to 10 detected genes for "All Cells" view

### Fixed
- `mitochondria_plot()` no longer modifies input AnnData in place; now uses a mask
- Intersection handling for user-provided cell lists with missing barcodes

## [0.2.8] 2025-12
### Added
- Doublet detection functionality using `Scrublet`, with an optional flag to remove detected doublets from retained cells after two-step cell calling.
- Mitochondrial percentage (`pct_counts_mt`) is added to `adata.obs` by default.

### NOTE:
- Metadata for doublet detection (Scrublet) and mitochondrial QC metrics are stored only when the input is an H5AD file.
- For matrix-based (MTX) inputs (e.g. legacy alevin-fry outputs), doublet removal and mitochondrial QC plotting are supported; however, these metadata are not written to additional text files.

### Modified
- Extend `.h5ad` output support to **mtx-based** inputs, enabling QCatch to save both the full and filtered .h5ad files for all input types (MTX and .h5ad, simpleaf v0.19.5+).
- For **mtx-based** input, save the intermediat results in `.h5ad`, No longer save to separate `.txt` files.

### Fix
- Logger set up.
- gene-symbol name duplicates for write mtx-based h5ad

## [0.2.7] 2025-11-04
### Added
- Support python 3.13.
### Changed

- Added automatic inference of the chemistry version from simpleaf quantification metadata.
- Removed the default chemistry assumption — QCatch now requires either:
    1. a successfully inferred chemistry from simpleaf’s metadata,
    2. an explicitly specified chemistry via --chemistry (-c), or
    3. a custom `number of partitions` provided via --n_partitions (-n).

    If none of these are supplied, QCatch will stop and prompt the user to specify one.

## [0.2.6] 2025-06-29

### Added

- Updated QCatch documentation and included an interactive demo page
- Add tutorial scripts in the README.
- Transitioned to uv for building and package management and relaxed dependencies for compatibility.

## [0.2.5] 2025-05-19

### Added

- Adopted Cookiecutter-style structure based on the Scanpy project template.
- Added a new flag to export summary metrics as a CSV file.
- The HTML report now also includes a warning for low mapping rate.
- Added unit tests and scripts to download test data
- Updated the EmptyDrops step by removing the limitation on the number of candidate barcodes and making the FDR threshold dynamically adjustable based on the chemistry version.
- Added source code snippets to the help text section of clustering plots

### Changed

- Switched to more concise progress logging during the cell-calling step.
