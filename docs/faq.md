## 🥐 Documentation for FeatureDump/ Column names in Anndata.obs
| **Term** | **Description** |
|------------|-----------------|
| **CB** | Cell Barcode |
| **Corrected Reads** | The total number of corrected reads, including both mapped and unmapped reads. |
| **Mapped Reads** | The number of reads that were successfully mapped to the reference. |
| **Deduplicated Reads** | The number of deduplicated reads, processed by UMI deduplication. |
| **Mapping Rate** | The ratio of mapped reads to corrected reads. |
| **Deduplication Rate** | The ratio of deduplicated reads to mapped reads. |
| **Mean by Max** | Calculated as mean_expr / max_umi, where:<br>mean_expr is the total UMI count divided by the number of genes with non-zero expression levels.<br>max_umi is the maximum expression level across all genes. |
| **Number of Genes Expressed** | The number of genes expressed in this cell. |
| **Number of Genes Above Mean** | The number of genes with expression levels above the mean_expr, where mean_expr is defined as the total UMI count divided by the number of genes with non-zero expression levels. |

## 🥝 Summary metric
| **Value showed in summary table** | **Description** |
|------------|-----------------|
| **Number of retained cells** | The number of valid and high quality cells that passed the cell calling step. This includes cells identified during the initial filtering and additional cells identified by the EmptyDrops step, whose expression profiles are significantly distinct from the ambient background. |
| **Number of all processed cells** | The total number of cell barcodes observed in the processed sample. Cells with zero reads have been excluded. |
| **Mean reads per retained cell** | The total number of reads assigned to the retained cells, including the mapped and unmapped reads, divided by the number of retained cells. |
| **Median UMI per retained cell** | The median number of deduplicated reads (UMIs) per retained cell. |
| **Median genes per retained cell** | The median number of genes detected per retained cell. |
| **Total genes detected for retained cells** | The total number of unique genes detected across all retained cells. |
| **Mapping rate** | Fraction of reads that mapped to the augmented reference, calculated as mapped reads / total processed reads. |
| **Sequencing saturation** | Sequencing saturation measures the proportion of reads coming from already-seen UMIs, calculated as 1 - (deduplicated reads / total reads). High saturation suggests limited gain from additional sequencing, while low saturation indicates that further sequencing could reveal more unique molecules (UMIs). |

## 🍰 3- Plots
#### 3.1  Knee plots
We order the UMI count(deduplicated reads) for each cell and sort in descending order to get the cell rank, and use the scatter plot to display the UMI count against the cell rank. We also use the scatter plot to display the number of detected genes against the cell rank.

#### 3.2 UMI Counts and Detected Gene Across Cell Barcodes
The barcode frequency is calculated as the number of reads associated with each cell barcode.

The **first two** plots show barcode frequency against two key metrics: the number of UMIs and the number of detected genes per barcode. The **third** plot illustrates how the number of detected genes increases with UMI count per cell.

#### 3.3 UMI Deduplication plot
The scatter plot compares the number of mapped reads and number of UMIs for each retained cell. Each point represents a cell, with the x-axis showing the mapped reads count and the y-axis showing the deduplicated UMIs count. The reference line indicates the mean deduplication rate across all cells.

**UMI Deduplication**: UMI deduplication is the process of identifying and removing duplicate reads that arise from PCR amplification of the same original molecule.

**Dedup Rate**: The UMI count divided by the number of mapped reads for each cell.

#### 3.4 Distribution of Detected Gene Count and Mitochondrial Percentage Plot
The plot depicts the distribution of detected gene counts per cell. The violin plot shows the distribution of mitochondrial gene expression percentages.

**Note**: The “All Cells” plot does not display every processed cell. To improve visualization and reduce clutter from very low-quality cells, we excluded cells with fewer than 20 detected genes—these are typically considered nearly empty. In contrast, the “Retained Cells” plot includes all retained cells, without applying this gene count filter.

#### 3.5 Bar Plot for S/U/A Counts and Splicing Ratio Distribution
When using “USA mode” in alevin-fry, spliced (S), unspliced (U), and ambiguous (A) read counts are generated separately for each gene in each cell.

In the **bar plot**, we first sum the spliced, unspliced, and ambiguous counts across all genes and all cells. The plot then displays the total number of reads in each splicing category: Spliced (S), Unspliced (U), and Ambiguous (A).

In the **histogram**, we calculate the splicing ratio for each cell as (S + A) / (S + U + A), where the counts are summed across all genes. The histogram shows the distribution of these per-cell splicing ratios.

#### 3.6 Clustering: UMAP and t-SNE
These plots represent low-dimensional projections of high-dimensional gene expression profiles. Each point corresponds to a single cell. Cells that are positioned close to one another in the plot are inferred to have similar transcriptomic signatures, suggesting similarity in cell type or state.

**Note**: Only retained cells are included in these visualizations, and no additional filtering was applied beyond the cell retention step. Standard preprocessing was performed using Scanpy, including normalization, log transformation, highly variable gene selection, and dimensionality reduction.
