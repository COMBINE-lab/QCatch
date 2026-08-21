"""
Regression tests for the numba-compiled simulate_multinomial_loglikelihoods.

The compiled implementation must be bit-identical to the reference pure-Python loop
(kept verbatim below, taken from QCatch v0.2.12 / nh3/emptydrops) for the same RNG state.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.stats as sp_stats

from qcatch.logger import setup_logger

setup_logger("qcatch", False)

from qcatch.find_retained_cells import cell_calling as cc  # noqa: E402
from qcatch.find_retained_cells.matrix import CountMatrix  # noqa: E402

TEST_H5AD = Path(__file__).resolve().parent / "data" / "test_data" / "simpleaf_with_map" / "quants.h5ad"


def _reference_simulate(profile_p, umis_per_bc, rng, num_sims=1000, jump=1000, n_sample_feature_block=1000000):
    """Verbatim copy of the pre-numba implementation (RNG passed explicitly instead of module global)."""
    distinct_n = np.flatnonzero(np.bincount(umis_per_bc.astype(int)))
    loglk = np.zeros((len(distinct_n), num_sims), dtype=float)
    sampled_features = rng.choice(len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True)
    k = 0
    log_profile_p = np.log(profile_p)
    for sim_idx in range(num_sims):
        curr_counts = np.ravel(sp_stats.multinomial.rvs(distinct_n[0], profile_p, size=1, random_state=rng))
        curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[0], p=profile_p)
        loglk[0, sim_idx] = curr_loglk
        for i in range(1, len(distinct_n)):
            step = distinct_n[i] - distinct_n[i - 1]
            if step >= jump:
                curr_counts += np.ravel(sp_stats.multinomial.rvs(step, profile_p, size=1, random_state=rng))
                curr_loglk = sp_stats.multinomial.logpmf(curr_counts, distinct_n[i], p=profile_p)
            else:
                for n in range(distinct_n[i - 1] + 1, distinct_n[i] + 1):
                    j = sampled_features[k]
                    k += 1
                    if k >= n_sample_feature_block:
                        sampled_features = rng.choice(
                            len(profile_p), size=n_sample_feature_block, p=profile_p, replace=True
                        )
                        k = 0
                    curr_counts[j] += 1
                    curr_loglk += log_profile_p[j] + np.log(float(n) / curr_counts[j])
            loglk[i, sim_idx] = curr_loglk
    return distinct_n, loglk


def _run_both(profile_p, umis, seed, **kw):
    rng = np.random.default_rng(seed)
    ref_n, ref_lk = _reference_simulate(profile_p, umis, rng, **kw)
    cc.RNG = np.random.default_rng(seed)
    new_n, new_lk = cc.simulate_multinomial_loglikelihoods(profile_p, umis, **kw)
    return ref_n, ref_lk, new_n, new_lk


def test_sim_bit_identical_synthetic_with_refill_and_jump():
    """Synthetic profile; small sample block so the refill path is exercised, plus two >= jump steps."""
    rng = np.random.default_rng(0)
    profile_p = rng.random(2000)
    profile_p /= profile_p.sum()
    # dense run of small steps, then two big jumps (>= jump=1000)
    umis = np.concatenate([np.arange(500, 1400, 2), [3000, 6000]])
    ref_n, ref_lk, new_n, new_lk = _run_both(
        profile_p, umis, seed=42, num_sims=40, jump=1000, n_sample_feature_block=5000
    )
    assert np.array_equal(ref_n, new_n)
    assert np.array_equal(ref_lk, new_lk)
    assert np.isfinite(new_lk).all()


def test_sim_bit_identical_default_block_size():
    """Default 1e6 sample block (no refill within the run)."""
    rng = np.random.default_rng(1)
    profile_p = rng.random(5000)
    profile_p /= profile_p.sum()
    umis = np.arange(500, 800)
    ref_n, ref_lk, new_n, new_lk = _run_both(profile_p, umis, seed=7, num_sims=10)
    assert np.array_equal(ref_lk, new_lk)


@pytest.mark.skipif(not TEST_H5AD.exists(), reason="test data not downloaded")
def test_sim_bit_identical_real_ambient_profile():
    """Real SGT ambient profile + real candidate UMI counts from the 1k PBMC test set."""
    import anndata as ad

    matrix = CountMatrix.from_anndata(ad.read_h5ad(TEST_H5AD))
    chem = "10X_3p_v3"
    filtered = cc.initial_filtering_OrdMag(matrix, chem, None)

    umis = matrix.get_counts_per_bc()
    bc_order = np.argsort(umis)
    lower, upper = cc.compute_empty_drops_bounds(chem, None)
    empty = bc_order[::-1][lower:upper]
    empty.sort()
    nz = np.flatnonzero(umis)
    use = np.intersect1d(empty, nz, assume_unique=True)
    _, profile_p = cc.est_background_profile_sgt(matrix.m, use)

    orig = {x.decode() if isinstance(x, bytes) else str(x) for x in filtered}
    is_orig = np.fromiter((bc.decode() in orig for bc in matrix.bcs), count=len(matrix.bcs), dtype=bool)
    min_umis = max(cc.MIN_UMIS, 1 + int(np.max(umis[empty], initial=0)))
    cand = umis[(~is_orig) & (umis >= min_umis)]
    assert len(cand) > 0

    ref_n, ref_lk, new_n, new_lk = _run_both(profile_p, cand, seed=42, num_sims=20)
    assert np.array_equal(ref_n, new_n)
    assert np.array_equal(ref_lk, new_lk)
