#!/usr/bin/env python
#
# Copyright (c) 2018 10X Genomics, Inc. All rights reserved.
from __future__ import annotations

import _io as io
import errno
import gzip
import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sp_io
import scipy.sparse as sp_sparse

from . import h5_constants
from .feature_ref import FeatureDef, FeatureReference

pd.set_option("compute.use_numexpr", False)
DEFAULT_DATA_DTYPE = "int32"
MATRIX_H5_VERSION = 2


# some helper functions from stats
def sum_sparse_matrix(matrix: sp_sparse.spmatrix, axis: int = 0) -> np.ndarray:
    """Sum a sparse matrix along an axis.

    Args:
        matrix (sp_sparse.spmatrix): Sparse matrix to sum.
        axis (int, optional): Axis along which to sum. Defaults to 0.

    Returns
    -------
        np.ndarray: Summed values as a numpy array.
    """
    return np.squeeze(np.asarray(matrix.sum(axis=axis)))


def makedirs(dst: str, allow_existing: bool = False) -> None:
    """Create a directory recursively. Optionally succeed if already exists.

    Useful because transient NFS server issues may induce double creation attempts.
    """
    if allow_existing:
        try:
            os.makedirs(dst)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(dst):
                pass
            else:
                raise
    else:
        os.makedirs(dst)


def open_maybe_gzip(filename: str | os.PathLike, mode: str = "r") -> io.BufferedIOBase:
    """Open a file normally or with compression based on file extension."""
    # this _must_ be a str
    filename = str(filename)
    if filename.endswith(h5_constants.GZIP_SUFFIX):
        raw = gzip.open(filename, mode + "b", 2)
    elif filename.endswith(h5_constants.LZ4_SUFFIX):
        import lz4

        raw = lz4.open(filename, mode + "b")
    else:
        return open(filename, mode)

    bufsize = 1024 * 1024  # 1MB of buffering
    if mode == "r":
        return io.BufferedReader(raw, buffer_size=bufsize)
    elif mode == "w":
        return io.BufferedWriter(raw, buffer_size=bufsize)
    else:
        raise ValueError(f"Unsupported mode for compression: {mode}")


def save_features_tsv(feature_ref: FeatureReference, base_dir: str, compress: bool, legacy: bool = True) -> None:
    """Save a FeatureReference to a tsv file"""
    if legacy:
        out_features_fn = os.path.join(base_dir, "genes.tsv")
        with open_maybe_gzip(out_features_fn, "w") as f:
            for feature_def in feature_ref.feature_defs:
                f.write("\t".join((feature_def.id, feature_def.name)) + "\n")
    else:
        out_features_fn = os.path.join(base_dir, "features.tsv")
        if compress:
            out_features_fn += ".gz"
        with open_maybe_gzip(out_features_fn, "w") as f:
            for feature_def in feature_ref.feature_defs:
                f.write(
                    ("\t".join((feature_def.id, feature_def.name, feature_def.feature_type)) + "\n").encode("ascii")
                )


class CountMatrix:
    """A sparse matrix wrapper storing gene expression counts with associated barcodes and features."""

    def __init__(self, feature_ref: FeatureReference, bcs: list[bytes], matrix: sp_sparse.spmatrix) -> None:
        """Initialize CountMatrix with features, barcodes, and matrix."""
        # Features (genes, CRISPR gRNAs, antibody barcodes, etc.)
        self.feature_ref = feature_ref
        self.features_dim = len(feature_ref.feature_defs)
        self.feature_ids_map = {f.id: f.index for f in feature_ref.feature_defs}

        # Cell barcodes
        bcs = np.array(bcs, dtype="S")
        bcs.flags.writeable = False
        self.bcs = bcs
        (self.bcs_dim,) = self.bcs.shape
        bcs_idx = np.argsort(self.bcs).astype(np.int32)
        bcs_idx.flags.writeable = False
        self.bcs_idx = bcs_idx

        self.m = matrix

    def get_shape(self) -> tuple[int, int]:
        """Return the shape of the sliced matrix"""
        return self.m.shape

    def get_num_nonzero(self) -> int:
        """Return the number of nonzero entries in the sliced matrix"""
        return self.m.nnz

    @classmethod
    def empty(cls, feature_ref: FeatureReference, bcs: list[bytes], dtype: str = DEFAULT_DATA_DTYPE) -> CountMatrix:
        """Create an empty matrix."""
        matrix = sp_sparse.lil_matrix((len(feature_ref.feature_defs), len(bcs)), dtype=dtype)
        return cls(feature_ref=feature_ref, bcs=bcs, matrix=matrix)

    @staticmethod
    def from_anndata(adata: Any) -> CountMatrix:
        """Create CountMatrix from anndata object."""
        barcodes = adata.obs["barcodes"]
        genes = adata.var_names.values
        feature_defs = [FeatureDef(idx, gene_id, None, "Gene Expression", []) for (idx, gene_id) in enumerate(genes)]
        feature_ref = FeatureReference(feature_defs, [])
        matrix = adata.X.T.astype(int)
        if type(matrix) is not sp_sparse.csc_matrix:
            matrix = matrix.tocsc()
        mat = CountMatrix(feature_ref, barcodes, matrix)
        return mat

    @staticmethod
    def from_legacy_mtx(genome_dir: str) -> CountMatrix:
        """Load CountMatrix from legacy matrix market format."""
        barcodes_tsv = os.path.join(genome_dir, "barcodes.tsv")
        genes_tsv = os.path.join(genome_dir, "genes.tsv")
        matrix_mtx = os.path.join(genome_dir, "matrix.mtx")
        for filepath in [barcodes_tsv, genes_tsv, matrix_mtx]:
            if not os.path.exists(filepath):
                raise OSError(f"Required file not found: {filepath}")
        barcodes = pd.read_csv(barcodes_tsv, delimiter="\t", header=None, usecols=[0]).values.squeeze()
        genes = pd.read_csv(genes_tsv, delimiter="\t", header=None, usecols=[0, 1], names=["gene_id", "name"])
        feature_defs = [
            FeatureDef(idx, row["gene_id"], row["name"], "Gene Expression", []) for idx, row in genes.iterrows()
        ]
        feature_ref = FeatureReference(feature_defs, [])

        matrix = sp_io.mmread(matrix_mtx)
        mat = CountMatrix(feature_ref, barcodes, matrix)
        mat.tocsc()
        return mat

    @staticmethod
    def from_v3_mtx(genome_dir: str) -> CountMatrix:
        """Load CountMatrix from v3 matrix market format."""
        barcodes_tsv = os.path.join(genome_dir, "barcodes.tsv.gz")
        features_tsv = os.path.join(genome_dir, "features.tsv.gz")
        matrix_mtx = os.path.join(genome_dir, "matrix.mtx.gz")
        for filepath in [barcodes_tsv, features_tsv, matrix_mtx]:
            if not os.path.exists(filepath):
                raise OSError(f"Required file not found: {filepath}")
        barcodes = pd.read_csv(barcodes_tsv, delimiter="\t", header=None, usecols=[0]).values.squeeze()
        features = pd.read_csv(features_tsv, delimiter="\t", header=None)

        feature_defs = []
        for idx, (_, r) in enumerate(features.iterrows()):
            fd = FeatureDef(idx, r[0], r[1], r[2], [])
            feature_defs.append(fd)

        feature_ref = FeatureReference(feature_defs, [])

        matrix = sp_io.mmread(matrix_mtx).tocsc()
        mat = CountMatrix(feature_ref, barcodes, matrix)
        return mat

    @staticmethod
    def load_mtx(mtx_dir: str) -> CountMatrix:
        """Load CountMatrix from matrix directory, choosing correct version.

        Args:
            mtx_dir (str): Directory containing matrix files.

        Raises
        ------
            OSError: If directory does not contain valid matrix files.

        Returns
        -------
            CountMatrix: Loaded CountMatrix instance.
        """
        legacy_fn = os.path.join(mtx_dir, "genes.tsv")
        v3_fn = os.path.join(mtx_dir, "features.tsv.gz")

        if os.path.exists(legacy_fn):
            return CountMatrix.from_legacy_mtx(mtx_dir)

        if os.path.exists(v3_fn):
            return CountMatrix.from_v3_mtx(mtx_dir)

        raise OSError(f"Not a valid path to a feature-barcode mtx directory: '{str(mtx_dir)}'")

    def tolil(self) -> None:
        """Convert matrix to LIL format if not already."""
        if type(self.m) is not sp_sparse.lil_matrix:
            self.m = self.m.tolil()

    def tocoo(self) -> None:
        """Convert matrix to COO format if not already."""
        if type(self.m) is not sp_sparse.coo_matrix:
            self.m = self.m.tocoo()

    def tocsc(self) -> None:
        """Convert matrix to CSC format if not already."""
        # Convert from lil to csc matrix for efficiency when analyzing data
        if type(self.m) is not sp_sparse.csc_matrix:
            self.m = self.m.tocsc()

    def select_barcodes(self, indices: list[int]) -> CountMatrix:
        """Select a subset of barcodes and return the resulting CountMatrix."""
        return CountMatrix(feature_ref=self.feature_ref, bcs=[self.bcs[i] for i in indices], matrix=self.m[:, indices])

    def select_barcodes_by_seq(self, barcode_seqs: list[bytes]) -> CountMatrix:
        """Select barcodes by their sequence."""
        return self.select_barcodes([self.bc_to_int(bc) for bc in barcode_seqs])

    def select_features(self, indices: list[int]) -> CountMatrix:
        """Select a subset of features and return the resulting matrix. We also update FeatureDefs to keep their indices consistent with their new position"""
        old_feature_defs = [self.feature_ref.feature_defs[i] for i in indices]

        updated_feature_defs = [
            FeatureDef(index=i, id=fd.id, name=fd.name, feature_type=fd.feature_type, tags=fd.tags)
            for (i, fd) in enumerate(old_feature_defs)
        ]

        feature_ref = FeatureReference(feature_defs=updated_feature_defs, all_tag_keys=self.feature_ref.all_tag_keys)

        return CountMatrix(feature_ref=feature_ref, bcs=self.bcs, matrix=self.m[indices, :])

    def select_features_by_ids(self, feature_ids: list[str]) -> CountMatrix:
        """Select features by their IDs.

        Args:
            feature_ids (list[str]): List of feature IDs.
        """
        return self.select_features(self.feature_ids_to_ints(feature_ids))

    def get_unique_features_per_bc(self) -> np.ndarray:
        """Get number of unique features per barcode."""
        return sum_sparse_matrix(self.m[self.m > 0], axis=0)

    def get_counts_per_bc(self) -> np.ndarray:
        """Get counts per barcode."""
        return sum_sparse_matrix(self.m, axis=0)

    def get_counts_per_feature(self) -> np.ndarray:
        """Get counts per feature."""
        return sum_sparse_matrix(self.m, axis=1)

    def get_numbcs_per_feature(self) -> np.ndarray:
        """Get number of barcodes per feature."""
        return sum_sparse_matrix(self.m > 0, axis=1)

    def get_top_bcs(self, cutoff: int) -> np.ndarray:
        """Get indices of top barcodes by count cutoff."""
        reads_per_bc = self.get_counts_per_bc()
        index = max(0, min(reads_per_bc.size, cutoff) - 1)
        value = sorted(reads_per_bc, reverse=True)[index]
        return np.nonzero(reads_per_bc >= value)[0]

    def int_to_bc(self, j: int) -> bytes:
        """Convert integer index to barcode sequence."""
        return self.bcs[j]

    def ints_to_bcs(self, jj: list[int]) -> list[bytes]:
        """Convert list of indices to barcode sequences."""
        return [self.int_to_bc(j) for j in jj]

    def save_mex(
        self,
        base_dir: str,
        save_features_func: Callable[[FeatureReference, str, bool, bool], None] = save_features_tsv,
        metadata: dict[str, Any] | None = None,
        compress: bool = True,
        legacy: bool = True,
    ) -> None:
        """Save in Matrix Market Exchange format."""
        self.tocoo()

        makedirs(base_dir, allow_existing=True)

        out_matrix_fn = os.path.join(base_dir, "matrix.mtx")
        out_barcodes_fn = os.path.join(base_dir, "barcodes.tsv")
        if compress:
            out_matrix_fn += ".gz"
            out_barcodes_fn += ".gz"

        # This method only supports an integer matrix.
        assert self.m.dtype in ["uint32", "int32", "uint64", "int64"]
        assert isinstance(self.m, sp_sparse.coo_matrix)

        rows, cols = self.m.shape
        # Header fields in the file
        rep = "coordinate"
        field = "integer"
        symmetry = "general"

        metadata = metadata or {}
        metadata.update(
            {
                "format_version": MATRIX_H5_VERSION,
            }
        )

        metadata_str = json.dumps(metadata)
        comment = "" if legacy else f"metadata_json: {metadata_str}"

        with open_maybe_gzip(out_matrix_fn, "w") as stream:
            # write initial header line
            stream.write(np.compat.asbytes(f"%%MatrixMarket matrix {rep} {field} {symmetry}\n"))

            # write comments
            for line in comment.split("\n"):
                stream.write(np.compat.asbytes(f"%{line}\n"))

            # write shape spec
            stream.write(np.compat.asbytes(f"{rows} {cols} {self.m.nnz}\n"))
            # write row, col, val in 1-based indexing
            for r, c, d in zip(self.m.row + 1, self.m.col + 1, self.m.data, strict=False):
                stream.write(np.compat.asbytes(f"{r} {c} {d}\n"))

        # both GEX and ATAC provide an implementation of this in respective feature_ref.py
        save_features_func(self.feature_ref, base_dir, compress=compress, legacy=legacy)

        with open_maybe_gzip(out_barcodes_fn, "w") as f:
            for bc in self.bcs:
                f.write(bc + b"\n")
