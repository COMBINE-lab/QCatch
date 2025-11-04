#!/usr/bin/env python

"""Functions for calling cell-associated barcodes"""

import logging
import os
import sys
from collections import namedtuple

import numpy as np
import numpy.ma as ma
import scipy.stats as sp_stats
from scipy.sparse import csc_matrix

from qcatch.logger import QCatchLogger

from .sgt import SimpleGoodTuringError, sgt_proportions

logger = logging.getLogger("qcatch")
assert isinstance(logger, QCatchLogger), "Logger is not a QCatchLogger. Call setup_logger() in main.py first."
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# #----------------- Constants -----------------
ORDMAG_RECOVERED_CELLS_QUANTILE = 0.99
ORDMAG_NUM_BOOTSTRAP_SAMPLES = 100
MIN_RECOVERED_CELLS = 5

MAX_RECOVERED_CELLS = 2621448  # 2^18
NP_SORT_KIND = "stable"

# Number of partitions (max number of barcodes to consider for ambient estimation)1
N_PARTITIONS = 90000

# Drop this top fraction of the barcodes when estimating ambient.
MAX_OCCUPIED_PARTITIONS_FRAC = 0.5

# *M* Minimum number of UMIS per barcode to consider after the initial cell calling
MIN_UMIS = 500

# Default number of background simulations to make
NUM_SIMS = 100000

# Minimum ratio of UMIs to the median (initial cell call UMI) to consider after the initial cell calling
MIN_UMI_FRAC_OF_MEDIAN = 0.01

MAX_MEM_GB = 0.3

NonAmbientBarcodeResult = namedtuple(
    "NonAmbientBarcodeResult",
    [
        "eval_bcs",  # Candidate barcode indices (n)
        "log_likelihood",  # Ambient log likelihoods (n)
        "pvalues",  # pvalues (n)
        "pvalues_adj",  # B-H adjusted pvalues (n)
        "is_nonambient",  # Boolean nonambient calls (n)
    ],
)


class FilteredCellResults:
    """Class to store results of filtered cell selection."""

    def __init__(self, value: int = 0) -> None:
        self.n_top_cells = value
        self.selection_cutoff = value


def compute_empty_drops_bounds(
    chemistry_description: str | None = None, n_partitions: int | None = None
) -> tuple[int, int]:
    """Determines the lower and upper bounds for empty drops background based on the provided chemistry description."""
    if n_partitions is not None:
        return (n_partitions // 2, n_partitions)

    if chemistry_description in ["10X_3p_v4", "10X_5p_v3", "10X_HT"]:
        n_partitions = 160000
    elif chemistry_description == "10X_3p_LT":
        n_partitions = 9000
    elif chemistry_description in ["10X_3p_v2", "10X_3p_v3"]:
        n_partitions = 90000
    else:
        # invalid chemistry description
        logger.error(f"Invalid chemistry description: {chemistry_description}. Exiting.")
        raise SystemExit(1)
    return (n_partitions // 2, n_partitions)


def get_fdr_threshold_by_chemistry(chemistry_name: str | None) -> float:
    """
    Return the maximum adjusted p-value (FDR threshold) for calling a barcode as non-ambient, based on the chemistry used.

    Parameters
    ----------
    chemistry_name
        Name of the 10X chemistry.

    Returns
    -------
    float
        FDR threshold (e.g., 0.001 or 0.01)
    """
    high_gem_chemistries = {"10X_3p_v3", "10X_3p_v4", "10X_5p_v3", "10X_HT"}
    if not chemistry_name:
        # chemistry is None or empty string
        logger.warning("Chemistry not specified. Using default FDR threshold (0.01).")
        return 0.01
    return 0.001 if chemistry_name in high_gem_chemistries else 0.01


def find_within_ordmag(resampled_bc_counts: np.ndarray, quantile_point: int) -> int:
    """
    Find the number of barcodes above a cutoff determined by a quantile value.

    Parameters
    ----------
    resampled_bc_counts
        Resampled barcode counts derived from non-zero_bc_counts.
    quantile_point
        Index of the cell at the quantile point.

    Returns
    -------
    int
        Number of barcodes above the cutoff (quantile_val / 10).
    """
    n = len(resampled_bc_counts)
    if n == 0:
        return 0
    sorted_bc_counts = np.sort(resampled_bc_counts)[::-1]
    # get the quantile value
    left = int(np.floor(quantile_point))
    right = int(np.ceil(quantile_point))

    if left == right:
        quantile_val = sorted_bc_counts[left]
    else:
        leftval = sorted_bc_counts[left]
        rightval = sorted_bc_counts[right]
        leftgap = quantile_point - left
        rightgap = right - quantile_point
        quantile_val = leftval * rightgap + rightval * leftgap

    # Compute cutoff
    cutoff = max(1, int(0.1 * quantile_val))

    num_cells_above_cutoff = np.searchsorted(-sorted_bc_counts, -cutoff, side="right")

    return num_cells_above_cutoff


def ordMag_expected(sorted_bc_counts: np.ndarray, recovered_cells: np.ndarray) -> tuple[int, float]:
    """
    Estimate the expected number of cells by analyzing the distribution of barcode counts.

    Parameters
    ----------
    sorted_bc_counts
        A sorted (descending) array of read counts per barcode.
    recovered_cells
        A list of candidate cell indices (log2-spaced) to evaluate potential cell thresholds.

    Returns
    -------
    tuple
        The estimated number of expected cells and the associated loss value.
    """
    # Initialize loss array
    loss = np.zeros(len(recovered_cells))
    filtered_cells = np.zeros(len(recovered_cells))
    cutoff_point = 0
    for idx, cell_idx in enumerate(recovered_cells):
        if cell_idx >= len(sorted_bc_counts):  # Ensure index is within bounds
            continue

        quantile_point = cell_idx * (1 - ORDMAG_RECOVERED_CELLS_QUANTILE)

        left = int(np.floor(quantile_point))
        right = int(np.ceil(quantile_point))

        if left == right:
            quantile_val = sorted_bc_counts[left]
        else:
            leftval = sorted_bc_counts[left]
            rightval = sorted_bc_counts[right]
            leftgap = quantile_point - left
            rightgap = right - quantile_point
            quantile_val = leftval * rightgap + rightval * leftgap

        # Finding the cutoff
        cutoff = 0.1 * quantile_val
        while cutoff_point < len(sorted_bc_counts) and sorted_bc_counts[cutoff_point] > cutoff:
            cutoff_point += 1

        # Number of cells is one less than the cutoff_point
        num_cells = cutoff_point - 1

        # Compute loss safely
        if num_cells > 0:
            loss[idx] = ((num_cells - cell_idx) ** 2) / cell_idx
        else:
            loss[idx] = np.inf  # Assign high loss if no valid cells
        filtered_cells[idx] = num_cells
        optimal_idx = np.argmin(loss)

    # Return the best estimate from log2-spaced search
    return recovered_cells[optimal_idx], loss[optimal_idx]


def call_ordMag(nonzero_bc_counts: np.ndarray, max_expected_cells: int) -> tuple[int, float]:
    """Call the expected number of cells using the OrdMag method."""
    sorted_bc_counts = np.sort(nonzero_bc_counts)[::-1]
    # Generate log2-spaced cell indices
    recovered_cells = np.linspace(1, np.log2(max_expected_cells), 2000)
    recovered_cells = np.unique(np.round(np.power(2, recovered_cells)).astype(int))

    return ordMag_expected(sorted_bc_counts, recovered_cells)


def compute_bootstrapped_top_n(top_n_boot: np.ndarray, nonzero_counts: np.ndarray) -> FilteredCellResults:
    """Compute the bootstrapped top N cells from bootstrap samples."""
    top_n_bcs_mean = np.mean(top_n_boot)
    n_top_cells = int(np.round(top_n_bcs_mean))
    logger.debug(f"INSIDE compute_bootstrapped_top_n(): n_top_cells: {n_top_cells}")
    result = FilteredCellResults()
    result.n_top_cells = n_top_cells

    if n_top_cells > 0:
        sorted_indices = np.argsort(nonzero_counts, kind=NP_SORT_KIND)[::-1]
        sorted_counts = nonzero_counts[sorted_indices]
        cutoff = sorted_counts[n_top_cells - 1]

        # Expand cells with same count as cutoff
        same_cutoff_indices = np.where(sorted_counts == cutoff)[0]
        logger.debug(f"same_cutoff_indices: {same_cutoff_indices}")
        expanded_index = same_cutoff_indices[-1] + 1

        # If expansion exceeds 20%, revert to a soft upper bound estimate
        if (expanded_index + 1 - n_top_cells) > 0.20 * n_top_cells:
            result.n_top_cells = int(np.floor(n_top_cells * 1.2))
            return result
        # Update result with expanded selection
        result.n_top_cells = expanded_index + 1
        result.selection_cutoff = cutoff
    return result


def initial_filtering_OrdMag(
    matrix: csc_matrix, chemistry_description: str | None = None, n_partitions: int | None = None, verbose: bool = False
) -> tuple[np.ndarray, FilteredCellResults, str] | np.ndarray:
    """Perform initial filtering of cells using the OrdMag method."""
    metrics = FilteredCellResults(0)
    bc_counts = matrix.get_counts_per_bc()
    nonzero_bc_counts = bc_counts[bc_counts > 0]
    logger.debug(f"nonzero_bc_counts len : {len(nonzero_bc_counts)}")
    if len(nonzero_bc_counts) == 0:
        msg = "WARNING: All barcodes do not have enough reads for ordmag, allowing no bcs through"
        logger.record_warning(msg)
        return []

    # Determine the maximum number of cells to examine based on the empty drops range.

    lower_bound, _ = compute_empty_drops_bounds(chemistry_description, n_partitions)
    max_expected_cells = min(lower_bound, MAX_RECOVERED_CELLS)
    logger.debug(f"max_expected_cells: {max_expected_cells}")
    # Initialize a reproducible random state.
    rs = np.random.RandomState(0)

    # Bootstrap sampling to estimate recovered cells and loss.
    bootstrap_results = [
        call_ordMag(rs.choice(nonzero_bc_counts, size=len(nonzero_bc_counts)), max_expected_cells)
        for _ in range(ORDMAG_NUM_BOOTSTRAP_SAMPLES)
    ]
    # Compute average recovered cells and loss across all bootstrap samples.
    avg_recovered_cells, avg_loss = np.mean(np.stack(bootstrap_results), axis=0)

    # Ensure the recovered cells meets the minimum requirement.
    recovered_cells = max(int(np.round(avg_recovered_cells)), MIN_RECOVERED_CELLS)
    logger.debug(f"Found recovered_cells = {recovered_cells} with loss = {avg_loss}")

    baseline_bc_idx = int(np.round(float(recovered_cells) * (1 - ORDMAG_RECOVERED_CELLS_QUANTILE)))
    baseline_bc_idx = min(baseline_bc_idx, len(nonzero_bc_counts) - 1)
    logger.debug(f"Baseline BC index: {baseline_bc_idx}")
    # Bootstrap sampling; run algo with many random samples of the data
    top_n_boot = np.array(
        [
            find_within_ordmag(rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), baseline_bc_idx)
            for _ in range(ORDMAG_NUM_BOOTSTRAP_SAMPLES)
        ]
    )

    metrics = compute_bootstrapped_top_n(top_n_boot, nonzero_bc_counts)

    # Get the filtered barcodes
    top_n = metrics.n_top_cells
    top_bc_idx = np.sort(np.argsort(bc_counts, kind=NP_SORT_KIND)[::-1][0:top_n])
    assert top_n <= len(nonzero_bc_counts), "Invalid selection of 0-count barcodes!"
    # Convert the indices to barcode strings
    filtered_bcs = matrix.ints_to_bcs(top_bc_idx)
    filtered_bcs = np.array(filtered_bcs)
    # check if initial cells has very few UMIs
    # Filter based on UMI count
    filtered_counts = bc_counts[top_bc_idx]
    # keep_mask = filtered_counts > MIN_UMIS
    keep_mask = filtered_counts > MIN_UMIS
    num_high_quality_filtered_bcs = len(filtered_counts[keep_mask])

    if num_high_quality_filtered_bcs < len(filtered_bcs):
        msg = f"⚠️ Warning❗️: During the initial cell calling step, OrdMag identified {len(filtered_bcs) - num_high_quality_filtered_bcs} cells with UMI counts below {MIN_UMIS}. This situation is uncommon and may indicate that the dataset is of poor quality ⚠️. It is recommended to exclude these low-quality cells by evaluating per-cell UMI counts from the count matrix before proceeding with downstream analysis."
        logger.record_warning(msg)
    return filtered_bcs


def adjust_pvalue_bh(p: np.ndarray) -> np.ndarray:
    """Multiple testing correction of p-values using the Benjamini-Hochberg procedure."""
    descending = np.argsort(p)[::-1]
    # q = p * N / k where p = p-value, N = # tests, k = p-value rank
    scale = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(scale * p[descending]))

    # Return to original order
    return q[np.argsort(descending)]


def eval_multinomial_loglikelihoods(matrix: csc_matrix, profile_p: np.ndarray, max_mem_gb: float = 0.1) -> np.ndarray:
    """Compute the multinomial log PMF for many barcodes."""
    gb_per_bc = float(matrix.shape[0] * matrix.dtype.itemsize) / (1024**3)
    bcs_per_chunk = max(1, int(round(max_mem_gb / gb_per_bc)))
    num_bcs = matrix.shape[1]

    loglk = np.zeros(num_bcs)

    for chunk_start in range(0, num_bcs, bcs_per_chunk):
        chunk = slice(chunk_start, chunk_start + bcs_per_chunk)
        matrix_chunk = matrix[:, chunk].transpose().toarray()
        n = matrix_chunk.sum(1)
        loglk[chunk] = sp_stats.multinomial.logpmf(matrix_chunk, n, p=profile_p)
    return loglk


def simulate_multinomial_loglikelihoods(
    profile_p: np.ndarray,
    umis_per_bc: np.ndarray,
    num_sims: int = 1000,
    jump: int = 1000,
    n_sample_feature_block: int = 1000000,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate draws from a multinomial distribution for various values of N.

    Uses the approximation from Lun et al. (https://www.biorxiv.org/content/biorxiv/early/2018/04/04/234872.full.pdf)
    """
    distinct_n = np.flatnonzero(np.bincount(umis_per_bc.astype(int)))

    loglk = np.zeros((len(distinct_n), num_sims), dtype=float)
    num_all_n = np.max(distinct_n) - np.min(distinct_n)

    if verbose:
        logger.debug("Number of distinct N supplied: %d", len(distinct_n))
        logger.debug("Range of N: %d", num_all_n)
        logger.debug("Number of features: %d", len(profile_p))

    sampled_features = np.random.choice(len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True)
    k = 0

    log_profile_p = np.log(profile_p)

    for sim_idx in range(num_sims):
        if verbose and sim_idx % 1000 == 999:
            logger.debug("Simulation progress: completed %d/%d simulations", sim_idx + 1, num_sims)
        curr_counts = np.ravel(sp_stats.multinomial.rvs(distinct_n[0], profile_p, size=1))

        curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[0], p=profile_p)

        loglk[0, sim_idx] = curr_loglk

        for i in range(1, len(distinct_n)):
            step = distinct_n[i] - distinct_n[i - 1]
            if step >= jump:
                # Instead of iterating for each n, sample the intermediate ns all at once
                curr_counts += np.ravel(sp_stats.multinomial.rvs(step, profile_p, size=1))
                curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[i], p=profile_p)
                assert not np.isnan(curr_loglk)
            else:
                # Iteratively sample between the two distinct values of n
                for n in range(distinct_n[i - 1] + 1, distinct_n[i] + 1):
                    j = sampled_features[k]
                    k += 1
                    if k >= n_sample_feature_block:
                        # Amortize this operation
                        sampled_features = np.random.choice(
                            len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True
                        )
                        k = 0
                    curr_counts[j] += 1
                    curr_loglk += log_profile_p[j] + np.log(float(n) / curr_counts[j])

            loglk[i, sim_idx] = curr_loglk

    return distinct_n, loglk


def compute_ambient_pvalues(
    umis_per_bc: np.ndarray, obs_loglk: np.ndarray, sim_n: np.ndarray, sim_loglk: np.ndarray
) -> np.ndarray:
    """Compute p-values for observed multinomial log-likelihoods."""
    assert len(umis_per_bc) == len(obs_loglk)
    assert sim_loglk.shape[0] == len(sim_n)

    # Find the index of the simulated N for each barcode
    sim_n_idx = np.searchsorted(sim_n, umis_per_bc)
    num_sims = sim_loglk.shape[1]

    num_barcodes = len(umis_per_bc)

    pvalues = np.zeros(num_barcodes)

    for i in range(num_barcodes):
        num_lower_loglk = np.sum(sim_loglk[sim_n_idx[i], :] < obs_loglk[i])
        pvalues[i] = float(1 + num_lower_loglk) / (1 + num_sims)
    return pvalues


def estimate_profile_sgt(matrix: csc_matrix, barcode_indices: np.ndarray, nz_feat: np.ndarray) -> np.ndarray:
    """Estimate a gene expression profile by Simple Good Turing."""
    # Initial profile estimate
    prof_mat = matrix[:, barcode_indices]

    profile = np.ravel(prof_mat[nz_feat, :].sum(axis=1))
    zero_feat = np.flatnonzero(profile == 0)

    # Simple Good Turing estimate
    p_smoothed, p0 = sgt_proportions(profile[np.flatnonzero(profile)])

    # Distribute p0 equally among the zero elements.
    p0_i = p0 / len(zero_feat)

    profile_p = np.repeat(p0_i, len(nz_feat))
    profile_p[np.flatnonzero(profile)] = p_smoothed

    assert np.isclose(profile_p.sum(), 1.0)
    return profile_p


# Construct a background expression profile from barcodes with <= T UMIs
def est_background_profile_sgt(matrix: csc_matrix, use_bcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a gene expression profile on a given subset of barcodes. Use Good-Turing to smooth the estimated profile."""
    # Use features that are nonzero anywhere in the data
    use_feats = np.flatnonzero(np.asarray(matrix.sum(1)))

    # Estimate background profile
    bg_profile_p = estimate_profile_sgt(matrix, use_bcs, use_feats)

    return (use_feats, bg_profile_p)


def find_nonambient_barcodes(
    matrix: csc_matrix,
    orig_cell_bcs: list[str],
    chemistry_description: str | None,
    n_partitions: int,
    max_mem_gb: float = MAX_MEM_GB,
    min_umi_frac_of_median: float = MIN_UMI_FRAC_OF_MEDIAN,
    min_umis_nonambient: int = MIN_UMIS,
    verbose: bool = False,
) -> NonAmbientBarcodeResult | None:
    """
    Call barcodes as being sufficiently distinct from the ambient profile.

    Parameters
    ----------
    matrix
        Full expression matrix.
    orig_cell_bcs
        Initially-called cell barcodes.
    chemistry_description
        Chemistry description.
    n_partitions
        Number of partitions used for ambient estimation.
    max_mem_gb
        Maximum memory to use in gigabytes.
    min_umi_frac_of_median
        Minimum UMI fraction (relative to median) required for candidate barcodes.
    min_umis_nonambient
        Minimum number of UMIs to consider a barcode for non-ambient testing.
    verbose
        Whether to enable verbose output.

    Returns
    -------
    NonAmbientBarcodeResult or None
        A result object containing evaluation metrics for non-ambient barcodes, or None if no suitable barcodes are found.
    """
    # Estimate an ambient RNA profile
    umis_per_bc = matrix.get_counts_per_bc()
    logger.debug(f"umi_per_bc zero count: {np.count_nonzero(umis_per_bc == 0)}")
    logger.debug(f" median of umis_per_bc: {np.median(umis_per_bc)}")
    bc_order = np.argsort(umis_per_bc)

    lower_bound, upper_bound = compute_empty_drops_bounds(chemistry_description, n_partitions)
    # MODIFIED
    logger.debug(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    # Take what we expect to be the barcodes associated w/ empty partitions.
    empty_bcs = bc_order[::-1][lower_bound:upper_bound]
    empty_bcs.sort()

    # Require non-zero barcodes
    nz_bcs = np.flatnonzero(umis_per_bc)
    nz_bcs.sort()

    logger.debug(f"len of non-zero counts: {len(empty_bcs)}")

    use_bcs = np.intersect1d(empty_bcs, nz_bcs, assume_unique=True)

    if len(use_bcs) > 0:
        try:
            eval_features, ambient_profile_p = est_background_profile_sgt(matrix.m, use_bcs)
        except SimpleGoodTuringError as e:
            logger.warning(str(e))
            return None
    else:
        eval_features = np.zeros(0, dtype=int)
        ambient_profile_p = np.zeros(0)

    # # Choose candidate cell barcodes
    orig_cell_bc_set = set(orig_cell_bcs)
    orig_cells = np.flatnonzero(
        np.fromiter((bc in orig_cell_bc_set for bc in matrix.bcs), count=len(matrix.bcs), dtype=bool)
    )
    logger.debug(f"len of orig_cells: {len(orig_cells)}")
    # No good incoming cell calls
    if orig_cells.sum() == 0:
        logger.warning("⚠️ No good incoming cell calls, exit early")
        return None

    # Look at non-cell barcodes above a minimum UMI count
    eval_bcs = np.ma.array(np.arange(matrix.bcs_dim))
    eval_bcs[orig_cells] = ma.masked

    # median_initial_umis = np.median(umis_per_bc[orig_cells])
    # min_umis = int(max(min_umis_nonambient, round(np.ceil(median_initial_umis * min_umi_frac_of_median))))
    # logger.debug('Median UMIs of initial cell calls: {}'.format(median_initial_umis))
    max_ambient_umis = np.max(umis_per_bc[empty_bcs], initial=0)
    logger.debug(f"Max UMIs of ambient barcodes: {max_ambient_umis}")

    min_umis = max(min_umis_nonambient, 1 + max_ambient_umis)
    logger.debug(f"Min UMIs: {min_umis}")

    eval_bcs[umis_per_bc < min_umis] = ma.masked
    n_unmasked_bcs = len(eval_bcs) - eval_bcs.mask.sum()
    # Take the unmasked barcodes
    eval_bcs = np.argsort(ma.masked_array(umis_per_bc, mask=eval_bcs.mask))[0:n_unmasked_bcs]

    if len(eval_bcs) == 0:
        return None

    assert not np.any(np.isin(eval_bcs, orig_cells))
    logger.debug(f"Number of candidate bcs: {len(eval_bcs)}")
    logger.debug(f"Range candidate bc umis: {umis_per_bc[eval_bcs].min()}, {umis_per_bc[eval_bcs].max()}")

    eval_mat = matrix.m[eval_features, :][:, eval_bcs]

    if len(ambient_profile_p) == 0:
        obs_loglk = np.repeat(np.nan, len(eval_bcs))
        pvalues = np.repeat(1, len(eval_bcs))
        sim_loglk = np.repeat(np.nan, len(eval_bcs))
        return None

    # Compute observed log-likelihood of barcodes being generated from ambient RNA
    logger.info(f"🍰 Starting eval_multinomial_loglikelihoods() using max_mem_gb = {max_mem_gb} GB")
    obs_loglk = eval_multinomial_loglikelihoods(eval_mat, ambient_profile_p, max_mem_gb=max_mem_gb)

    # Simulate log likelihoods
    distinct_ns, sim_loglk = simulate_multinomial_loglikelihoods(
        ambient_profile_p, umis_per_bc[eval_bcs], num_sims=10000, verbose=verbose
    )

    # Compute p-values
    pvalues = compute_ambient_pvalues(umis_per_bc[eval_bcs], obs_loglk, distinct_ns, sim_loglk)

    pvalues_adj = adjust_pvalue_bh(pvalues)
    # get the max_adj by chemistry
    max_adj_pvalue = get_fdr_threshold_by_chemistry(chemistry_description)
    is_nonambient = pvalues_adj <= max_adj_pvalue
    eval_bcs_str = matrix.ints_to_bcs(eval_bcs)
    # converted the byte strings to regular strings
    eval_bcs = np.array([x.decode() for x in eval_bcs_str])

    return NonAmbientBarcodeResult(
        eval_bcs=eval_bcs,
        log_likelihood=obs_loglk,
        pvalues=pvalues,
        pvalues_adj=pvalues_adj,
        is_nonambient=is_nonambient,
    )
