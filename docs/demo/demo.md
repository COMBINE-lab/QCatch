### Demo report

In this demo, we applied QCatch to a publicly available human peripheral blood mononuclear cell (PBMC) scRNA-seq dataset obtained from the 10X Genomics website.

This [dataset](https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0). comprises scRNA-seq data from approximately 1,000 PBMCs isolated from a healthy human donor and generated using the Chromium Single Cell 3′ Gene Expression v3 chemistry.

#### Demo 1: Original dataset

In the first demo, we applied QCatch to the original dataset to demonstrate the standard QCatch report generated from a high-quality input. This example illustrates the typical QC metrics and visualizations produced when sufficient sequencing depth and data quality are available.

#### Demo 2: Low-depth dataset

To demonstrate QCatch’s behavior on low-depth sequencing data, we generated a reduced-depth dataset by subsampling 20% of reads from the original FASTQ files. This subsampling resulted in a substantial decrease in sequencing depth, with the mean reads per cell reduced from 46,957 to 9,750.

When applying QCatch to this low-depth dataset, no additional non-ambient (true) cells were rescued beyond the initial cell-calling step, indicating poor data quality. As expected, this scenario triggered a warning message in the QCatch report (highlighted in yellow).
