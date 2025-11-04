# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

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
