#!/usr/bin/env python

""" Functions for calling cell-associated barcodes """
import os 
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from collections import namedtuple
import numpy as np
import numpy.ma as ma
import scipy.stats as sp_stats
from .sgt import sgt_proportions,SimpleGoodTuringError
import time
import logging
logger = logging.getLogger(__name__)
# #----------------- Constants -----------------
ORDMAG_RECOVERED_CELLS_QUANTILE = 0.99
ORDMAG_NUM_BOOTSTRAP_SAMPLES = 100
MIN_RECOVERED_CELLS = 5

MAX_RECOVERED_CELLS = 2621448 # 2^18
NP_SORT_KIND = "stable"

# Number of additional barcodes to consider after the initial cell calling
N_CANDIDATE_BARCODES=20000

# Number of partitions (max number of barcodes to consider for ambient estimation)
N_PARTITIONS=90000

# Drop this top fraction of the barcodes when estimating ambient.
MAX_OCCUPIED_PARTITIONS_FRAC = 0.5

# *M* Minimum number of UMIS per barcode to consider after the initial cell calling
MIN_UMIS = 500

# Default number of background simulations to make
NUM_SIMS = 100000

# Minimum ratio of UMIs to the median (initial cell call UMI) to consider after the initial cell calling
MIN_UMI_FRAC_OF_MEDIAN = 0.01

# *M* Maximum adjusted p-value to call a barcode as non-ambient
MAX_ADJ_PVALUE = 0.01

MAX_MEM_GB = 0.3 

NonAmbientBarcodeResult = namedtuple('NonAmbientBarcodeResult',
                                     ['eval_bcs',      # Candidate barcode indices (n)
                                      'log_likelihood',# Ambient log likelihoods (n)
                                      'pvalues',       # pvalues (n)
                                      'pvalues_adj',   # B-H adjusted pvalues (n)
                                      'is_nonambient', # Boolean nonambient calls (n)
                                      ])

## *M*
class FilteredCellResults:
    def __init__(self, value=0):
        self.n_top_cells = value
        self.selection_cutoff = value

## *M*
def compute_empty_drops_bounds(chemistry_description: str | None = None, n_partitions: int | None = None):
    """Determines the lower and upper bounds for empty drops background based on the provided chemistry description.

    Args:
        chemistry_description: A string specifying the chemistry type.
        Number of partitions specified in the input.

    Returns:
        tuple(lower_bounds, upper_bounds)
    """
    if n_partitions is not None:
        return (n_partitions // 2, n_partitions)
    
    if chemistry_description in ["10X_3p_v4", "10X_5p_v3","10X_HT"]:
        n_partitions = 160000
    elif chemistry_description == "10X_3p_LT":
        n_partitions = 9000
    elif chemistry_description in ["10X_3p_v2", "10X_3p_v3"]:
        n_partitions = 90000
        # TODO: logging info
    else:
        n_partitions = 90000
    return (n_partitions // 2, n_partitions)

## *M*
def find_within_ordmag(resampled_bc_counts, quantile_point):
    """
        Args:
        resampled_bc_counts (np.ndarray): resampled barcode counts derived from non-zero_bc_counts.
        quantile_point (int): index of the cell at the quantile point.
        
        Returns:
        int: The number of barcodes above cutoff (quantile_val / 10).
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
    
    num_cells_above_cutoff = np.searchsorted(-sorted_bc_counts, -cutoff, side='right')
    
    return num_cells_above_cutoff

## *M*
def ordMag_expected(sorted_bc_counts,recovered_cells ):
    """
    This is a Python implementation of the ordMagExpected function, which estimates the expected number of cells by analyzing the distribution of barcode counts in a sorted (descending) array. It iterates through 
    candidate cell indices (log2-spaced) to determine the optimal threshold for selecting cells.
    
    Parameters:
    sorted_bc_counts (array): A sorted (descending) array of read counts per barcode.
    recovered_cells (array): 
            A list of candidate cell indices(log2-spaced) to evaluate potential cell thresholds.

    Returns:
    int: The estimated number of expected cells.
    """
    # Initialize loss array
    loss = np.zeros(len(recovered_cells))
    filtered_cells = np.zeros(len(recovered_cells))
    cutoff_point = 0
    for idx, cell_idx in enumerate(recovered_cells):
        if cell_idx >= len(sorted_bc_counts):  # Ensure index is within bounds
            continue

        quantile_point = cell_idx  * (1-ORDMAG_RECOVERED_CELLS_QUANTILE)
        
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
            loss[idx] = ((num_cells - cell_idx ) ** 2) / cell_idx
        else:
            loss[idx] = np.inf  # Assign high loss if no valid cells
        filtered_cells[idx] = num_cells
        optimal_idx = np.argmin(loss)
        
    # Return the best estimate from log2-spaced search
    return recovered_cells[optimal_idx], loss[optimal_idx] 

## *M*
def call_ordMag(nonzero_bc_counts, max_expected_cells):
    sorted_bc_counts = np.sort(nonzero_bc_counts)[::-1]
    # Generate log2-spaced cell indices
    recovered_cells = np.linspace(1, np.log2(max_expected_cells), 2000)
    recovered_cells = np.unique(np.round(np.power(2, recovered_cells)).astype(int))

    return ordMag_expected(sorted_bc_counts, recovered_cells)

## *M*
def compute_bootstrapped_top_n(top_n_boot, nonzero_counts):
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
        expanded_index = same_cutoff_indices[-1]+1

        # If expansion exceeds 20%, revert to a soft upper bound estimate
        if (expanded_index + 1 - n_top_cells) > 0.20 * n_top_cells:
            result.n_top_cells = int(np.floor(n_top_cells * 1.2))
            return result
        # Update result with expanded selection
        result.n_top_cells = expanded_index + 1
        result.selection_cutoff = cutoff
    return result

## *M*
def initial_filtering_OrdMag(matrix, chemistry_description: str | None = None,n_partitions: int | None = None):
    
    metrics = FilteredCellResults(0)
    
    bc_counts = matrix.get_counts_per_bc()
    
    nonzero_bc_counts = bc_counts[bc_counts > 0]
    if len(nonzero_bc_counts) == 0:
        msg = "WARNING: All barcodes do not have enough reads for ordmag, allowing no bcs through"
        return [], metrics, msg
    
    # Determine the maximum number of cells to examine based on the empty drops range.
    if chemistry_description is None and n_partitions is None:
        max_expected_cells = MAX_RECOVERED_CELLS
    else:
        lower_bound, _ = compute_empty_drops_bounds(chemistry_description, n_partitions)
        max_expected_cells = min(lower_bound, MAX_RECOVERED_CELLS)

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
    
    
    baseline_bc_idx = int(np.round(float(recovered_cells) * (1-ORDMAG_RECOVERED_CELLS_QUANTILE)))
    baseline_bc_idx = min(baseline_bc_idx, len(nonzero_bc_counts) - 1)
    logger.debug(f"Baseline BC index: {baseline_bc_idx}")
    # Bootstrap sampling; run algo with many random samples of the data
    top_n_boot = np.array(
        [
            find_within_ordmag(
                rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), baseline_bc_idx
            ) 
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
    
    return filtered_bcs


def adjust_pvalue_bh(p):
    """ Multiple testing correction of p-values using the Benjamini-Hochberg procedure """
    descending = np.argsort(p)[::-1]
    # q = p * N / k where p = p-value, N = # tests, k = p-value rank
    scale = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(scale * p[descending]))

    # Return to original order
    return q[np.argsort(descending)]


def eval_multinomial_loglikelihoods(matrix, profile_p, max_mem_gb=0.1):
    """Compute the multinomial log PMF for many barcodes
    Args:
      matrix (scipy.sparse.csc_matrix): Matrix of UMI counts (feature x barcode)
      profile_p (np.ndarray(float)): Multinomial probability vector
      max_mem_gb (float): Try to bound memory usage.
    Returns:
      log_likelihoods (np.ndarray(float)): Log-likelihood for each barcode
    """
    gb_per_bc = float(matrix.shape[0] * matrix.dtype.itemsize) / (1024**3)
    bcs_per_chunk = max(1, int(round(max_mem_gb/gb_per_bc)))
    num_bcs = matrix.shape[1]

    loglk = np.zeros(num_bcs)

    for chunk_start in range(0, num_bcs, bcs_per_chunk):
        chunk = slice(chunk_start, chunk_start+bcs_per_chunk)
        matrix_chunk = matrix[:,chunk].transpose().toarray()
        n = matrix_chunk.sum(1)
        loglk[chunk] = sp_stats.multinomial.logpmf(matrix_chunk, n, p=profile_p)
    return loglk


def simulate_multinomial_loglikelihoods(profile_p, umis_per_bc,
                                        num_sims=1000, jump=1000,
                                        n_sample_feature_block=1000000, verbose=False):
    """Simulate draws from a multinomial distribution for various values of N.

       Uses the approximation from Lun et al. ( https://www.biorxiv.org/content/biorxiv/early/2018/04/04/234872.full.pdf )

    Args:
      profile_p (np.ndarray(float)): Probability of observing each feature.
      umis_per_bc (np.ndarray(int)): UMI counts per barcode (multinomial N).
      num_sims (int): Number of simulations per distinct N value.
      jump (int): Vectorize the sampling if the gap between two distinct Ns exceeds this.
      n_sample_feature_block (int): Vectorize this many feature samplings at a time.
    Returns:
      (distinct_ns (np.ndarray(int)), log_likelihoods (np.ndarray(float)):
      distinct_ns is an array containing the distinct N values that were simulated.
      log_likelihoods is a len(distinct_ns) x num_sims matrix containing the
        simulated log likelihoods.
    """
    distinct_n = np.flatnonzero(np.bincount(umis_per_bc.astype(int)))

    loglk = np.zeros((len(distinct_n), num_sims), dtype=float)
    num_all_n = np.max(distinct_n) - np.min(distinct_n)
    if verbose:
        print('Number of distinct N supplied: %d' % len(distinct_n))
        print('Range of N: %d' % num_all_n)
        print('Number of features: %d' % len(profile_p))

    sampled_features = np.random.choice(len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True)
    k = 0

    log_profile_p = np.log(profile_p)

    for sim_idx in range(num_sims):
        if verbose and sim_idx % 100 == 99:
            sys.stdout.write('.')
            sys.stdout.flush()
        curr_counts = np.ravel(sp_stats.multinomial.rvs(distinct_n[0], profile_p, size=1))

        curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[0], p=profile_p)

        loglk[0, sim_idx] = curr_loglk

        for i in range(1, len(distinct_n)):
            step = distinct_n[i] - distinct_n[i-1]
            if step >= jump:
                # Instead of iterating for each n, sample the intermediate ns all at once
                curr_counts += np.ravel(sp_stats.multinomial.rvs(step, profile_p, size=1))
                curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[i], p=profile_p)
                assert not np.isnan(curr_loglk)
            else:
                # Iteratively sample between the two distinct values of n
                for n in range(distinct_n[i-1]+1, distinct_n[i]+1):
                    j = sampled_features[k]
                    k += 1
                    if k >= n_sample_feature_block:
                        # Amortize this operation
                        sampled_features = np.random.choice(len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True)
                        k = 0
                    curr_counts[j] += 1
                    curr_loglk += log_profile_p[j] + np.log(float(n)/curr_counts[j])

            loglk[i, sim_idx] = curr_loglk

    if verbose:
        sys.stdout.write('\n')

    return distinct_n, loglk


def compute_ambient_pvalues(umis_per_bc, obs_loglk, sim_n, sim_loglk):
    """Compute p-values for observed multinomial log-likelihoods
    Args:
      umis_per_bc (nd.array(int)): UMI counts per barcode
      obs_loglk (nd.array(float)): Observed log-likelihoods of each barcode deriving from an ambient profile
      sim_n (nd.array(int)): Multinomial N for simulated log-likelihoods
      sim_loglk (nd.array(float)): Simulated log-likelihoods of shape (len(sim_n), num_simulations)
    Returns:
      pvalues (nd.array(float)): p-values
    """
    assert len(umis_per_bc) == len(obs_loglk)
    assert sim_loglk.shape[0] == len(sim_n)

    # Find the index of the simulated N for each barcode
    sim_n_idx = np.searchsorted(sim_n, umis_per_bc)
    num_sims = sim_loglk.shape[1]

    num_barcodes = len(umis_per_bc)

    pvalues = np.zeros(num_barcodes)

    for i in range(num_barcodes):
        num_lower_loglk = np.sum(sim_loglk[sim_n_idx[i],:] < obs_loglk[i])
        pvalues[i] = float(1 + num_lower_loglk) / (1 + num_sims)
    return pvalues

def estimate_profile_sgt(matrix, barcode_indices, nz_feat):
    """ Estimate a gene expression profile by Simple Good Turing.
    Args:
      raw_mat (sparse matrix): Sparse matrix of all counts
      barcode_indices (np.array(int)): Barcode indices to use
      nz_feat (np.array(int)): Indices of features that are non-zero at least once
    Returns:
      profile (np.array(float)): Estimated probabilities of length len(nz_feat).
    """
    # Initial profile estimate
    prof_mat = matrix[:,barcode_indices]

    profile = np.ravel(prof_mat[nz_feat, :].sum(axis=1))
    zero_feat = np.flatnonzero(profile == 0)

    # Simple Good Turing estimate
    p_smoothed, p0 = sgt_proportions(profile[np.flatnonzero(profile)])

    # Distribute p0 equally among the zero elements.
    p0_i = p0/len(zero_feat)

    profile_p = np.repeat(p0_i, len(nz_feat))
    profile_p[np.flatnonzero(profile)] = p_smoothed

    assert np.isclose(profile_p.sum(), 1.0)
    return profile_p


# Construct a background expression profile from barcodes with <= T UMIs
def est_background_profile_sgt(matrix, use_bcs):
    """ Estimate a gene expression profile on a given subset of barcodes.
         Use Good-Turing to smooth the estimated profile.
    Args:
      matrix (scipy.sparse.csc_matrix): Sparse matrix of all counts
      use_bcs (np.array(int)): Indices of barcodes to use (col indices into matrix)
    Returns:
      profile (use_features, np.array(float)): Estimated probabilities of length use_features.
    """
    # Use features that are nonzero anywhere in the data
    use_feats = np.flatnonzero(np.asarray(matrix.sum(1)))

    # Estimate background profile
    bg_profile_p = estimate_profile_sgt(matrix, use_bcs, use_feats)

    return (use_feats, bg_profile_p)


def find_nonambient_barcodes(matrix, orig_cell_bcs,chemistry_description, n_partitions, max_mem_gb = MAX_MEM_GB, min_umi_frac_of_median=MIN_UMI_FRAC_OF_MEDIAN,
                             min_umis_nonambient=MIN_UMIS,
                             max_adj_pvalue=MAX_ADJ_PVALUE,verbose=False):
    """ Call barcodes as being sufficiently distinct from the ambient profile

    Args:
      matrix (CountMatrix): Full expression matrix.
      orig_cell_bcs (iterable of str): Strings of initially-called cell barcodes.
    Returns:
    TBD
    """
    # Estimate an ambient RNA profile
    umis_per_bc = matrix.get_counts_per_bc()
    logger.debug(f'umi_per_bc zero count: {np.count_nonzero(umis_per_bc == 0)}')
    logger.debug(f' median of umis_per_bc: {np.median(umis_per_bc)}')
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

    logger.debug(f'len of non-zero counts: {len(empty_bcs)}')
    
    use_bcs = np.intersect1d(empty_bcs, nz_bcs, assume_unique=True)

    if len(use_bcs) > 0:
        try:
            eval_features, ambient_profile_p = est_background_profile_sgt(matrix.m, use_bcs)
        except SimpleGoodTuringError as e:
            print(str(e))
            return None
    else:
        eval_features = np.zeros(0, dtype=int)
        ambient_profile_p = np.zeros(0)

    # # Choose candidate cell barcodes
    orig_cell_bc_set = set(orig_cell_bcs)
    orig_cells = np.flatnonzero(np.fromiter((bc in orig_cell_bc_set for bc in matrix.bcs),
                                            count=len(matrix.bcs), dtype=bool))
    logger.debug(f'len of orig_cells: {len(orig_cells)}')
    # No good incoming cell calls
    if orig_cells.sum() == 0:
        logger.warning('⚠️ No good incoming cell calls, exit early')
        return None

    # Look at non-cell barcodes above a minimum UMI count
    eval_bcs = np.ma.array(np.arange(matrix.bcs_dim))
    eval_bcs[orig_cells] = ma.masked

    median_initial_umis = np.median(umis_per_bc[orig_cells])
    
    min_umis = int(max(min_umis_nonambient, round(np.ceil(median_initial_umis * min_umi_frac_of_median))))
    logger.debug('Median UMIs of initial cell calls: {}'.format(median_initial_umis))
    logger.debug('Min UMIs: {}'.format(min_umis))

    eval_bcs[umis_per_bc < min_umis] = ma.masked
    n_unmasked_bcs = len(eval_bcs) - eval_bcs.mask.sum()

    # Take the top N_CANDIDATE_BARCODES by UMI count, of barcodes that pass the above criteria
    eval_bcs = np.argsort(ma.masked_array(umis_per_bc, mask=eval_bcs.mask))[0:n_unmasked_bcs][-N_CANDIDATE_BARCODES:]

    if len(eval_bcs) == 0:
        return None

    assert not np.any(np.isin(eval_bcs, orig_cells))
    logger.debug('Number of candidate bcs: {}'.format(len(eval_bcs)))
    logger.debug('Range candidate bc umis: {}, {}'.format(umis_per_bc[eval_bcs].min(), umis_per_bc[eval_bcs].max()))

    eval_mat = matrix.m[eval_features, :][:, eval_bcs]

    if len(ambient_profile_p) == 0:
        obs_loglk = np.repeat(np.nan, len(eval_bcs))
        pvalues = np.repeat(1, len(eval_bcs))
        sim_loglk = np.repeat(np.nan, len(eval_bcs))
        return None

    # Compute observed log-likelihood of barcodes being generated from ambient RNA
    logger.info(f"🍰 Starting eval_multinomial_loglikelihoods() using max_mem_gb = {max_mem_gb} GB")
    obs_loglk = eval_multinomial_loglikelihoods(eval_mat, ambient_profile_p,max_mem_gb = max_mem_gb)

    # Simulate log likelihoods
    distinct_ns, sim_loglk = simulate_multinomial_loglikelihoods(ambient_profile_p, umis_per_bc[eval_bcs], num_sims=10000, verbose=verbose)

    # Compute p-values
    pvalues = compute_ambient_pvalues(umis_per_bc[eval_bcs], obs_loglk, distinct_ns, sim_loglk)

    pvalues_adj = adjust_pvalue_bh(pvalues)

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
