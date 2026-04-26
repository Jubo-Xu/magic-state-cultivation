"""
test_incremental_rref_batch.py — Hypothesis-based property tests for IncrementalRREFBatch.

Two sections:

Section 1 — Single-column equivalence (mirrors test_incremental_rref.py)
  For every layer in the original test suite, we replicate the test using
  IncrementalRREFBatch with single-element add_columns([h], s_extra) calls.
  Key assertion: results are field-for-field identical to IncrementalRREF
  (same T, U, H, s, s_prime, pivot_map, Z) because single-element add_columns
  applies the exact same row operations in the exact same order as add_column.

Section 2 — Multi-column batch behaviour
  Tests specific to add_columns with k > 1 columns simultaneously.
  Because the batch propagates each column j's row operations in-place into
  M'[:, j+1:] (via M'[l, j+1:] ^= M'[pivot_row, j+1:]), the sequence of
  row operations applied to T, U, and s_prime is mathematically identical to
  sequential add_column calls in the same order.  All fields remain
  field-identical to sequential, which the tests verify.

  2a: Full-batch equivalence   — add_columns(all_k) == sequential.
  2b: Random batch grouping    — columns in variable-sized batches == sequential.
  2c: Null space capture       — null vectors correct for mixed independent/dependent.
  2d: Block-diagonal optimisation — all-new-check-row batch still correct.
  2e: Batch merge              — IncrementalRREFBatch.merge with k > 1 connecting
                                 edges == IncrementalRREF.merge (sequential edges).
  2f: Varying-length columns   — zero-padding of shorter columns verified.

Usage
-----
    cd algorithms && pytest test_python/test_incremental_rref_batch.py -v
"""

import sys
import os

# Add algorithms/ to the import path so source modules are found.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from incremental_rref import IncrementalRREF, _gf2_rank
from incremental_rref_batch import IncrementalRREFBatch


# ===========================================================================
# Shared helpers
# ===========================================================================

def build_single_cluster(H: np.ndarray, s: np.ndarray,
                          frs: bool = False) -> IncrementalRREF:
    """
    Sequential IncrementalRREF reference: add all columns of H one by one.
    Column 0 establishes all n_checks rows via s_extra=s; the rest are
    same-length (no new check rows).
    """
    cluster = IncrementalRREF(free_region_store=frs)
    cluster.add_column(H[:, 0], s_extra=s)
    for j in range(1, H.shape[1]):
        cluster.add_column(H[:, j])
    return cluster


def build_single_cluster_batch(H: np.ndarray, s: np.ndarray,
                                frs: bool = False) -> IncrementalRREFBatch:
    """
    IncrementalRREFBatch using single-element add_columns([h], s_extra) calls.
    Mirrors build_single_cluster; should produce field-identical results.
    """
    cluster = IncrementalRREFBatch(free_region_store=frs)
    cluster.add_columns([H[:, 0]], s_extra=s)
    for j in range(1, H.shape[1]):
        cluster.add_columns([H[:, j]])
    return cluster


def _assert_fields_equal(batch_c, ref_c, label: str = '') -> None:
    """
    Assert that an IncrementalRREFBatch state is field-for-field identical to
    an IncrementalRREF state: n_checks, n_bits, H, T, U, s, s_prime,
    pivot_map, and Z (same count, same vectors in same order).
    """
    tag = f" [{label}]" if label else ""
    assert batch_c.n_checks == ref_c.n_checks,   f"n_checks mismatch{tag}"
    assert batch_c.n_bits   == ref_c.n_bits,     f"n_bits mismatch{tag}"
    assert np.array_equal(batch_c.H,       ref_c.H),       f"H mismatch{tag}"
    assert np.array_equal(batch_c.T,       ref_c.T),       f"T mismatch{tag}"
    assert np.array_equal(batch_c.U,       ref_c.U),       f"U mismatch{tag}"
    assert np.array_equal(batch_c.s,       ref_c.s),       f"s mismatch{tag}"
    assert np.array_equal(batch_c.s_prime, ref_c.s_prime), f"s_prime mismatch{tag}"
    assert batch_c.pivot_map == ref_c.pivot_map,           f"pivot_map mismatch{tag}"
    assert len(batch_c.Z) == len(ref_c.Z),                 f"|Z| mismatch{tag}"
    for k, (zb, zr) in enumerate(zip(batch_c.Z, ref_c.Z)):
        assert np.array_equal(zb, zr), f"Z[{k}] mismatch{tag}"


def _assert_invariants(c, label: str = '') -> None:
    """
    Verify the core output invariants without comparing to a reference:
    verify() passes, rank-nullity holds, every z in Z satisfies H @ z == 0.
    """
    c.verify()
    n_pivots = sum(1 for p in c.pivot_map if p is not None)
    tag = f" [{label}]" if label else ""
    assert n_pivots + len(c.Z) == c.n_bits, f"rank-nullity{tag}"
    for k, z in enumerate(c.Z):
        assert np.array_equal(
            (c.H @ z) % 2, np.zeros(c.n_checks, dtype=np.uint8)
        ), f"null vector {k} violates H @ z == 0{tag}"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def small_gf2_matrix_and_syndrome(draw, max_rows=8, max_cols=8):
    n_rows = draw(st.integers(1, max_rows))
    n_cols = draw(st.integers(1, max_cols))
    H = np.array(
        draw(st.lists(
            st.lists(st.integers(0, 1), min_size=n_cols, max_size=n_cols),
            min_size=n_rows, max_size=n_rows,
        )),
        dtype=np.uint8,
    )
    s = np.array(
        draw(st.lists(st.integers(0, 1), min_size=n_rows, max_size=n_rows)),
        dtype=np.uint8,
    )
    return H, s


@st.composite
def two_gf2_matrices_and_syndromes(draw, max_rows=6, max_cols=6):
    H1, s1 = draw(small_gf2_matrix_and_syndrome(max_rows=max_rows, max_cols=max_cols))
    H2, s2 = draw(small_gf2_matrix_and_syndrome(max_rows=max_rows, max_cols=max_cols))
    return H1, s1, H2, s2


# ===========================================================================
# Section 1 — Single-column equivalence
# ===========================================================================

class TestSection1_S1a_SameCheckColumns:
    """
    Mirror of Layer 1a: add only same-check columns.

    add_columns([h]) must give field-identical results to add_column(h) at
    every intermediate step.  verify() is called after every insert in both
    paths to ensure all six internal invariants hold throughout.
    """

    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=10))
    @settings(max_examples=200)
    def test_s1a_field_identical_after_each_insert(self, hs):
        H, s = hs
        for frs in (False, True):
            ref   = IncrementalRREF(free_region_store=frs)
            batch = IncrementalRREFBatch(free_region_store=frs)

            ref.add_column(H[:, 0], s_extra=s, verify=True)
            batch.add_columns([H[:, 0]], s_extra=s, verify=True)
            _assert_fields_equal(batch, ref, label=f"frs={frs} col=0")

            for j in range(1, H.shape[1]):
                ref.add_column(H[:, j], verify=True)
                batch.add_columns([H[:, j]], verify=True)
                _assert_fields_equal(batch, ref, label=f"frs={frs} col={j}")


class TestSection1_S1b_BlockStructure:
    """
    Mirror of Layer 1b: each column introduces exactly one new check row.

    Single-element add_columns must give field-identical results, and rank
    must grow by 1 at every step (block structure guarantees independence).
    """

    @given(
        n=st.integers(1, 10),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=100)
    def test_s1b_block_structure_field_identical(self, n, seed):
        rng = np.random.default_rng(seed)
        for frs in (False, True):
            ref   = IncrementalRREF(free_region_store=frs)
            batch = IncrementalRREFBatch(free_region_store=frs)
            rank_before = 0

            for j in range(n):
                h       = np.zeros(j + 1, dtype=np.uint8)
                h[:j]   = rng.integers(0, 2, size=j, dtype=np.uint8)
                h[j]    = 1
                s_extra = rng.integers(0, 2, size=1, dtype=np.uint8)

                ref.add_column(h, s_extra=s_extra, verify=True)
                batch.add_columns([h.copy()], s_extra=s_extra.copy(), verify=True)
                _assert_fields_equal(batch, ref, label=f"frs={frs} step={j}")

                rank_after = sum(1 for p in batch.pivot_map if p is not None)
                assert rank_after == rank_before + 1, (
                    f"Rank did not grow at step {j}: before={rank_before}, after={rank_after}"
                )
                rank_before = rank_after


class TestSection1_S1c_MixedInserts:
    """
    Mirror of Layer 1c: interleaved block-structure and same-check columns.

    Single-element add_columns must give field-identical results and rank-nullity
    must hold at every intermediate step.
    """

    @given(small_gf2_matrix_and_syndrome(max_rows=6, max_cols=12))
    @settings(max_examples=150)
    def test_s1c_mixed_inserts_field_identical(self, hs):
        H, s = hs
        n_checks, n_bits = H.shape
        for frs in (False, True):
            ref   = IncrementalRREF(free_region_store=frs)
            batch = IncrementalRREFBatch(free_region_store=frs)

            for j in range(n_bits):
                if j < n_checks:
                    h, s_ex = H[:j + 1, j], s[j:j + 1]
                else:
                    h, s_ex = H[:, j], None

                ref.add_column(h, s_extra=s_ex)
                batch.add_columns(
                    [h.copy()],
                    s_extra=(s_ex.copy() if s_ex is not None else None),
                )
                _assert_fields_equal(batch, ref, label=f"frs={frs} col={j}")

                n_pivots = sum(1 for p in batch.pivot_map if p is not None)
                assert n_pivots + len(batch.Z) == batch.n_bits


class TestSection1_S2_IsValid:
    """
    Mirror of Layer 2: is_valid() returns the same answer for batch vs
    sequential for both valid and invalid syndromes.
    """

    @given(small_gf2_matrix_and_syndrome(max_rows=7, max_cols=7))
    @settings(max_examples=200)
    def test_s2_valid_syndrome(self, hs):
        """s = H @ e (valid) — batch and sequential must agree and both return True."""
        H, _ = hs
        n_bits = H.shape[1]
        rng = np.random.default_rng(42)
        e = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        s = (H @ e) % 2

        for frs in (False, True):
            ref   = build_single_cluster(H, s, frs)
            batch = build_single_cluster_batch(H, s, frs)
            _assert_fields_equal(batch, ref)
            assert batch.is_valid(naive=False) == ref.is_valid(naive=False)
            assert batch.is_valid(naive=True, s_naive=s) == ref.is_valid(naive=True, s_naive=s)
            assert batch.is_valid(naive=False)

    @given(small_gf2_matrix_and_syndrome(max_rows=7, max_cols=5))
    @settings(max_examples=200)
    def test_s2_invalid_syndrome(self, hs):
        """Constructed invalid syndrome — batch and sequential must both return False."""
        H, _ = hs
        n_checks, n_bits = H.shape

        # Probe to find a zero row in the RREF.
        probe = IncrementalRREF()
        probe.add_column(H[:, 0], s_extra=np.zeros(n_checks, dtype=np.uint8))
        for j in range(1, n_bits):
            probe.add_column(H[:, j])

        zero_rows = [i for i in range(n_checks) if probe.pivot_map[i] is None]
        assume(len(zero_rows) > 0)

        def _gf2_solve(T, b):
            n = T.shape[0]
            aug = np.hstack([T.copy(), b.reshape(-1, 1)]).astype(np.uint8)
            for col in range(n):
                piv = next((r for r in range(col, n) if aug[r, col]), None)
                assert piv is not None, "T singular"
                aug[[col, piv]] = aug[[piv, col]]
                for row in range(n):
                    if row != col and aug[row, col]:
                        aug[row] ^= aug[col]
            return aug[:, n] % 2

        target          = np.zeros(n_checks, dtype=np.uint8)
        target[zero_rows[0]] = 1
        s_invalid       = _gf2_solve(probe.T, target)

        for frs in (False, True):
            ref   = build_single_cluster(H, s_invalid, frs)
            batch = build_single_cluster_batch(H, s_invalid, frs)
            _assert_fields_equal(batch, ref)
            assert batch.is_valid(naive=False) == ref.is_valid(naive=False)
            assert not batch.is_valid(naive=False)


class TestSection1_S3_Merge:
    """
    Mirror of Layer 3: IncrementalRREFBatch.merge must give field-identical
    results to IncrementalRREF.merge for any number of connecting edges.
    """

    @given(two_gf2_matrices_and_syndromes(max_rows=5, max_cols=5))
    @settings(max_examples=150)
    def test_s3_block_diagonal_merge(self, data):
        """No connecting edges: batch merge is field-identical to original merge."""
        H1, s1, H2, s2 = data
        for frs in (False, True):
            c1  = build_single_cluster(H1, s1, frs)
            c2  = build_single_cluster(H2, s2, frs)
            ref = IncrementalRREF.merge(c1, c2, connecting_edges=[], verify=True)

            c1b   = build_single_cluster_batch(H1, s1, frs)
            c2b   = build_single_cluster_batch(H2, s2, frs)
            batch = IncrementalRREFBatch.merge(c1b, c2b, connecting_edges=[], verify=True)

            _assert_fields_equal(batch, ref, label=f"frs={frs}")

    @given(
        two_gf2_matrices_and_syndromes(max_rows=4, max_cols=4),
        st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=100)
    def test_s3_connecting_edges(self, data, seed):
        """With connecting edges: batch merge is field-identical to original merge."""
        H1, s1, H2, s2 = data
        n_total = H1.shape[0] + H2.shape[0]
        rng = np.random.default_rng(seed)
        n_ce = int(rng.integers(0, 5))
        connecting_edges = [
            rng.integers(0, 2, size=n_total, dtype=np.uint8)
            for _ in range(n_ce)
        ]
        for frs in (False, True):
            c1  = build_single_cluster(H1, s1, frs)
            c2  = build_single_cluster(H2, s2, frs)
            ref = IncrementalRREF.merge(c1, c2,
                                        connecting_edges=connecting_edges,
                                        verify=True)

            c1b   = build_single_cluster_batch(H1, s1, frs)
            c2b   = build_single_cluster_batch(H2, s2, frs)
            batch = IncrementalRREFBatch.merge(c1b, c2b,
                                               connecting_edges=connecting_edges,
                                               verify=True)

            _assert_fields_equal(batch, ref, label=f"frs={frs}")


class TestSection1_S4_KnownCodes:
    """
    Mirror of Layer 4: known-code deterministic tests with IncrementalRREFBatch.
    Field-identical to IncrementalRREF; exact rank and null-space verified.
    """

    def test_s4_repetition_code(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        s = np.array([1, 0], dtype=np.uint8)
        for frs in (False, True):
            ref   = build_single_cluster(H, s, frs)
            batch = build_single_cluster_batch(H, s, frs)
            _assert_fields_equal(batch, ref)
            assert sum(1 for p in batch.pivot_map if p is not None) == 2
            assert len(batch.Z) == 1
            assert batch.is_valid(naive=False)

    def test_s4_full_rank_pcm(self):
        H = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=np.uint8)
        assert _gf2_rank(H) == 3
        s = np.array([1, 0, 1], dtype=np.uint8)
        for frs in (False, True):
            ref   = build_single_cluster(H, s, frs)
            batch = build_single_cluster_batch(H, s, frs)
            _assert_fields_equal(batch, ref)
            assert sum(1 for p in batch.pivot_map if p is not None) == 3
            assert len(batch.Z) == 0

    def test_s4_rank_deficient_valid_and_invalid(self):
        H        = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        s_inv    = np.array([1, 0, 0], dtype=np.uint8)
        s_valid  = np.array([1, 0, 1], dtype=np.uint8)
        for frs in (False, True):
            ref_inv   = build_single_cluster(H, s_inv, frs)
            batch_inv = build_single_cluster_batch(H, s_inv, frs)
            _assert_fields_equal(batch_inv, ref_inv)
            assert not batch_inv.is_valid(naive=False)

            ref_val   = build_single_cluster(H, s_valid, frs)
            batch_val = build_single_cluster_batch(H, s_valid, frs)
            _assert_fields_equal(batch_val, ref_val)
            assert batch_val.is_valid(naive=False)

    def test_s4_merge_two_repetition_codes(self):
        H  = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        s1 = np.array([1, 0], dtype=np.uint8)
        s2 = np.array([0, 1], dtype=np.uint8)
        for frs in (False, True):
            c1  = build_single_cluster(H, s1, frs)
            c2  = build_single_cluster(H, s2, frs)
            ref = IncrementalRREF.merge(c1, c2, connecting_edges=[], verify=True)

            c1b   = build_single_cluster_batch(H, s1, frs)
            c2b   = build_single_cluster_batch(H, s2, frs)
            batch = IncrementalRREFBatch.merge(c1b, c2b, connecting_edges=[], verify=True)
            _assert_fields_equal(batch, ref)

            assert sum(1 for p in batch.pivot_map if p is not None) == 4
            assert len(batch.Z) == 2


class TestSection1_S5_ColumnOrder:
    """
    Mirror of Layer 5: rank/|Z| column-order independence; _gf2_rank
    agreement; verify() at every intermediate state.
    """

    @given(
        small_gf2_matrix_and_syndrome(max_rows=6, max_cols=8),
        st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=150)
    def test_s5_column_order_independence(self, hs, seed):
        H, s = hs
        n_bits = H.shape[1]
        rng  = np.random.default_rng(seed)
        perm = rng.permutation(n_bits)
        for frs in (False, True):
            orig = build_single_cluster_batch(H, s, frs)
            perm_c = IncrementalRREFBatch(free_region_store=frs)
            perm_c.add_columns([H[:, perm[0]]], s_extra=s)
            for k in range(1, n_bits):
                perm_c.add_columns([H[:, perm[k]]])

            rank_orig = sum(1 for p in orig.pivot_map  if p is not None)
            rank_perm = sum(1 for p in perm_c.pivot_map if p is not None)
            assert rank_orig == rank_perm
            assert len(orig.Z) == len(perm_c.Z)

    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=8))
    @settings(max_examples=200)
    def test_s5_gf2_rank_agrees_with_pivot_count(self, hs):
        H, s = hs
        batch = build_single_cluster_batch(H, s)
        expected = _gf2_rank(H)
        actual   = sum(1 for p in batch.pivot_map if p is not None)
        assert actual == expected


# ===========================================================================
# Section 2 — Multi-column batch behaviour
# ===========================================================================

class TestSection2_FullBatchEquivalence:
    """
    2a: add_columns(all_k_columns_at_once) gives field-identical results to
    k sequential add_column calls.

    Why this holds: the batch propagates column j's row operations into
    M'[:, j+1:] in-place, so column j+1 sees exactly T_j @ h_{j+1} — the
    same value sequential would compute.  All row operations on T, U, and
    s_prime proceed identically.  Z vectors are also identical because
    pre-padding n_cols zeros and then placing z[n_bits+j]=1 is equivalent
    to sequential single-padding and then appending.
    """

    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=10))
    @settings(max_examples=300)
    def test_2a_all_at_once_field_identical(self, hs):
        """All columns added in a single add_columns call; field-identical to sequential."""
        H, s = hs
        for frs in (False, True):
            ref   = build_single_cluster(H, s, frs)
            batch = IncrementalRREFBatch(free_region_store=frs)
            batch.add_columns(list(H.T), s_extra=s, verify=True)
            _assert_fields_equal(batch, ref, label=f"frs={frs}")

    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=10))
    @settings(max_examples=200)
    def test_2a_is_valid_after_full_batch(self, hs):
        """is_valid() matches sequential for both naive and incremental modes."""
        H, s = hs
        ref   = build_single_cluster(H, s)
        batch = IncrementalRREFBatch()
        batch.add_columns(list(H.T), s_extra=s)
        assert batch.is_valid(naive=False) == ref.is_valid(naive=False)
        assert (batch.is_valid(naive=True, s_naive=s)
                == ref.is_valid(naive=True, s_naive=s))


class TestSection2_RandomBatchGrouping:
    """
    2b: Columns partitioned into random-sized contiguous batches; each batch
    added via one add_columns call.  The full sequence of column operations is
    unchanged, so results must be field-identical to sequential add_column.
    """

    @given(
        small_gf2_matrix_and_syndrome(max_rows=8, max_cols=12),
        st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=250)
    def test_2b_random_contiguous_batches_field_identical(self, hs, seed):
        """
        Split n_bits columns into 1..min(n_bits,5) contiguous groups of random
        sizes.  Each group is one add_columns call.  Result must match sequential.
        """
        H, s = hs
        n_bits = H.shape[1]
        rng = np.random.default_rng(seed)

        # Build a random contiguous partition.
        n_groups = int(rng.integers(1, min(n_bits + 1, 6)))
        if n_groups > 1 and n_bits > 1:
            cuts = sorted(rng.choice(range(1, n_bits), size=n_groups - 1,
                                     replace=False).tolist())
        else:
            cuts = []
        boundaries = [0] + cuts + [n_bits]
        groups = [list(range(boundaries[i], boundaries[i + 1]))
                  for i in range(len(boundaries) - 1)
                  if boundaries[i] < boundaries[i + 1]]

        for frs in (False, True):
            # Sequential reference: add columns in the flat order defined by groups.
            ref   = IncrementalRREF(free_region_store=frs)
            first = True
            for g in groups:
                for j in g:
                    ref.add_column(H[:, j], s_extra=(s if first else None))
                    first = False

            # Batch: one add_columns call per group, s_extra only for first group.
            batch = IncrementalRREFBatch(free_region_store=frs)
            first = True
            for g in groups:
                cols = [H[:, j] for j in g]
                batch.add_columns(cols, s_extra=(s if first else None), verify=True)
                first = False

            _assert_fields_equal(batch, ref, label=f"frs={frs}")


class TestSection2_NullSpaceCapture:
    """
    2c: When a batch contains both independent and dependent columns, every
    null vector z produced must satisfy H @ z == 0, and rank-nullity holds.
    Hypothesis tests complement explicit deterministic cases.
    """

    @given(small_gf2_matrix_and_syndrome(max_rows=7, max_cols=9))
    @settings(max_examples=200)
    def test_2c_null_vectors_satisfy_H_z_eq_0(self, hs):
        """Full-batch insert: verify() and H @ z == 0 for each z in Z."""
        H, s = hs
        batch = IncrementalRREFBatch()
        batch.add_columns(list(H.T), s_extra=s, verify=True)
        _assert_invariants(batch, label="full-batch")

    def test_2c_explicit_dependent_column_in_batch(self):
        """
        col0 = [1,0]^T, col1 = [0,1]^T, col2 = [1,0]^T (= col0, dependent).
        Rank = 2, null space dim = 1.  Batch and sequential field-identical.
        """
        H = np.array([[1, 0, 1],
                      [0, 1, 0]], dtype=np.uint8)
        s = np.array([1, 1], dtype=np.uint8)
        ref   = build_single_cluster(H, s)
        batch = IncrementalRREFBatch()
        batch.add_columns(list(H.T), s_extra=s, verify=True)
        _assert_fields_equal(batch, ref)
        assert len(batch.Z) == 1
        z = batch.Z[0]
        assert np.array_equal((H @ z) % 2, np.zeros(2, dtype=np.uint8))

    def test_2c_all_columns_identical_gives_rank_1(self):
        """
        H has rank 1 (all columns equal [1,1,1]^T).
        Full batch must produce n_bits - 1 null vectors, each satisfying H@z=0.
        """
        n_bits = 5
        H = np.ones((3, n_bits), dtype=np.uint8)
        s = np.array([1, 1, 1], dtype=np.uint8)
        batch = IncrementalRREFBatch()
        batch.add_columns(list(H.T), s_extra=s, verify=True)
        n_pivots = sum(1 for p in batch.pivot_map if p is not None)
        assert n_pivots == 1, f"Expected rank 1, got {n_pivots}"
        assert len(batch.Z) == n_bits - 1
        for z in batch.Z:
            assert np.array_equal((H @ z) % 2, np.zeros(3, dtype=np.uint8))

    def test_2c_alternating_independent_dependent(self):
        """
        Columns: [1,0],[0,1],[1,0],[0,1] — cols 2 and 3 are dependent on 0 and 1.
        Rank = 2, |Z| = 2.  Batch matches sequential field-for-field.
        """
        H = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1]], dtype=np.uint8)
        s = np.array([1, 0], dtype=np.uint8)
        ref   = build_single_cluster(H, s)
        batch = IncrementalRREFBatch()
        batch.add_columns(list(H.T), s_extra=s, verify=True)
        _assert_fields_equal(batch, ref)
        assert len(batch.Z) == 2
        for z in batch.Z:
            assert np.array_equal((H @ z) % 2, np.zeros(2, dtype=np.uint8))


class TestSection2_BlockDiagonalOpt:
    """
    2d: When all columns in a batch have zeros in every existing check row
    (M'[:n_old,:] == 0), the block-diagonal optimisation sets pivot_scan_start
    = n_old, skipping the guaranteed-zero upper block.  Results must still
    be field-identical to sequential.

    Construction: start with an n_old-row cluster; add a batch of columns
    whose upper n_old entries are all zero.
    """

    @given(
        n_old=st.integers(1, 6),
        n_new=st.integers(1, 6),
        n_cols=st.integers(1, 6),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=150)
    def test_2d_all_new_check_rows_field_identical(self, n_old, n_new, n_cols, seed):
        """
        After establishing n_old check rows, add a batch where every column
        is zero in rows 0..n_old-1.  The block-diagonal opt fires; result
        must be field-identical to sequential.
        """
        rng     = np.random.default_rng(seed)
        n_total = n_old + n_new

        # Initial cluster: n_old columns, each with exactly the n_old check rows.
        H_init  = rng.integers(0, 2, size=(n_old, n_old), dtype=np.uint8)
        s_init  = rng.integers(0, 2, size=n_old,          dtype=np.uint8)

        # Batch columns: zero in existing rows, random in new rows.
        H_batch = np.vstack([
            np.zeros((n_old, n_cols), dtype=np.uint8),           # upper block = 0
            rng.integers(0, 2, size=(n_new, n_cols), dtype=np.uint8),
        ])                                                         # shape (n_total, n_cols)
        s_extra = rng.integers(0, 2, size=n_new, dtype=np.uint8)

        for frs in (False, True):
            # Sequential reference.
            ref = build_single_cluster(H_init, s_init, frs)
            ref.add_column(H_batch[:, 0], s_extra=s_extra)
            for j in range(1, n_cols):
                ref.add_column(H_batch[:, j])

            # Batch: initial cluster (single-element add_columns) then one batch call.
            batch = build_single_cluster_batch(H_init, s_init, frs)
            batch.add_columns(list(H_batch.T), s_extra=s_extra, verify=True)

            _assert_fields_equal(batch, ref, label=f"frs={frs}")

    def test_2d_explicit_block_diagonal(self):
        """
        Deterministic: 2 existing check rows, batch of 3 columns that only
        touch 2 new check rows.  Verify opt fires and result is correct.
        """
        H_init  = np.eye(2, dtype=np.uint8)
        s_init  = np.array([1, 0], dtype=np.uint8)

        H_batch = np.array([
            [0, 0, 0],   # row 0 — existing, all-zero
            [0, 0, 0],   # row 1 — existing, all-zero
            [1, 0, 1],   # row 2 — new
            [0, 1, 1],   # row 3 — new
        ], dtype=np.uint8)
        s_extra = np.array([1, 1], dtype=np.uint8)

        ref = IncrementalRREF()
        ref.add_column(H_init[:, 0], s_extra=s_init)
        ref.add_column(H_init[:, 1])
        ref.add_column(H_batch[:, 0], s_extra=s_extra)
        ref.add_column(H_batch[:, 1])
        ref.add_column(H_batch[:, 2])

        batch = IncrementalRREFBatch()
        batch.add_columns([H_init[:, 0]], s_extra=s_init)
        batch.add_columns([H_init[:, 1]])
        batch.add_columns(list(H_batch.T), s_extra=s_extra, verify=True)

        _assert_fields_equal(batch, ref)


class TestSection2_BatchMerge:
    """
    2e: IncrementalRREFBatch.merge with k > 1 connecting edges must give
    field-identical results to IncrementalRREF.merge (which uses sequential
    add_column internally for each edge).
    """

    @given(
        two_gf2_matrices_and_syndromes(max_rows=4, max_cols=4),
        st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=150)
    def test_2e_batch_merge_field_identical(self, data, seed):
        """
        IncrementalRREFBatch.merge(c1, c2, [h0,..,hk-1]) is field-identical to
        IncrementalRREF.merge(c1, c2, [h0,..,hk-1]) for any k >= 0.
        """
        H1, s1, H2, s2 = data
        n_total = H1.shape[0] + H2.shape[0]
        rng     = np.random.default_rng(seed)
        n_ce    = int(rng.integers(0, 6))
        connecting_edges = [
            rng.integers(0, 2, size=n_total, dtype=np.uint8)
            for _ in range(n_ce)
        ]
        for frs in (False, True):
            c1  = build_single_cluster(H1, s1, frs)
            c2  = build_single_cluster(H2, s2, frs)
            ref = IncrementalRREF.merge(c1, c2,
                                        connecting_edges=connecting_edges,
                                        verify=True)

            c1b   = build_single_cluster_batch(H1, s1, frs)
            c2b   = build_single_cluster_batch(H2, s2, frs)
            batch = IncrementalRREFBatch.merge(c1b, c2b,
                                               connecting_edges=connecting_edges,
                                               verify=True)

            _assert_fields_equal(batch, ref, label=f"frs={frs} k={n_ce}")

    def test_2e_deterministic_three_connecting_edges(self):
        """
        Merge two known clusters with 3 explicit connecting edges.
        Exact n_bits and rank verified against IncrementalRREF reference.
        """
        H1 = np.array([[1, 1, 0],
                       [0, 1, 1]], dtype=np.uint8)
        H2 = np.array([[1, 0],
                       [0, 1]], dtype=np.uint8)
        s1 = np.array([1, 0], dtype=np.uint8)
        s2 = np.array([0, 1], dtype=np.uint8)

        ce = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1], dtype=np.uint8),
            np.array([1, 1, 0, 0], dtype=np.uint8),
        ]
        for frs in (False, True):
            c1  = build_single_cluster(H1, s1, frs)
            c2  = build_single_cluster(H2, s2, frs)
            ref = IncrementalRREF.merge(c1, c2, connecting_edges=ce, verify=True)

            c1b   = build_single_cluster_batch(H1, s1, frs)
            c2b   = build_single_cluster_batch(H2, s2, frs)
            batch = IncrementalRREFBatch.merge(c1b, c2b,
                                               connecting_edges=ce, verify=True)

            _assert_fields_equal(batch, ref, label=f"frs={frs}")
            assert batch.n_checks == 4
            assert batch.n_bits == 8     # b1=3, b2=2, 3 connecting


class TestSection2_VaryingLengthColumns:
    """
    2f: Columns of different lengths in the same add_columns call.

    The longest column determines n_new; shorter columns are zero-padded
    internally in add_columns.  The sequential reference manually zero-pads
    shorter columns before passing them to add_column.  Both must agree
    field-for-field.
    """

    @given(
        n_existing=st.integers(1, 5),
        n_new=st.integers(1, 5),
        n_cols=st.integers(2, 6),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=150)
    def test_2f_varying_length_zero_padding_field_identical(self,
                                                             n_existing, n_new,
                                                             n_cols, seed):
        """
        Start with n_existing check rows.  Add a batch of n_cols columns with
        lengths drawn from [n_existing, n_existing + n_new], with at least
        one having the maximum length n_existing + n_new (guaranteed by
        setting lengths[0] = n_existing + n_new).

        The sequential reference adds the longest column first (introducing
        n_new new check rows), then manually zero-pads shorter columns.
        add_columns zero-pads internally.  Both must give identical fields.
        """
        rng     = np.random.default_rng(seed)
        n_total = n_existing + n_new

        H_init  = rng.integers(0, 2, size=(n_existing, n_existing), dtype=np.uint8)
        s_init  = rng.integers(0, 2, size=n_existing,               dtype=np.uint8)

        # Column lengths in [n_existing, n_total]; lengths[0] = n_total (longest).
        lengths    = rng.integers(n_existing, n_total + 1, size=n_cols).tolist()
        lengths[0] = n_total
        cols_raw   = [
            rng.integers(0, 2, size=int(l), dtype=np.uint8) for l in lengths
        ]
        s_extra = rng.integers(0, 2, size=n_new, dtype=np.uint8)

        for frs in (False, True):
            # Sequential reference: add initial cluster, then the longest column
            # (which introduces n_new new rows), then remaining columns zero-padded.
            ref = build_single_cluster(H_init, s_init, frs)
            ref.add_column(cols_raw[0], s_extra=s_extra)
            for j in range(1, n_cols):
                col_padded = np.zeros(n_total, dtype=np.uint8)
                col_padded[:len(cols_raw[j])] = cols_raw[j]
                ref.add_column(col_padded)

            # Batch: pass all columns with their original lengths; add_columns
            # zero-pads shorter ones internally.
            batch = build_single_cluster_batch(H_init, s_init, frs)
            batch.add_columns(cols_raw, s_extra=s_extra, verify=True)

            _assert_fields_equal(batch, ref, label=f"frs={frs}")

    def test_2f_explicit_three_varying_length_columns(self):
        """
        Deterministic: 2 existing check rows; batch of 3 columns with lengths
        4, 3, 2 (n_existing=2, n_new=2).  Field-identical to sequential.

          col0 (len 4): [1,0,1,1] — touches all 4 rows
          col1 (len 3): [0,1,0]   — touches rows 0,1,2; row 3 = 0 after padding
          col2 (len 2): [1,1]     — touches rows 0,1; rows 2,3 = 0 after padding
        """
        H_init  = np.eye(2, dtype=np.uint8)
        s_init  = np.array([1, 0], dtype=np.uint8)

        col0    = np.array([1, 0, 1, 1], dtype=np.uint8)
        col1    = np.array([0, 1, 0],    dtype=np.uint8)
        col2    = np.array([1, 1],       dtype=np.uint8)
        s_extra = np.array([1, 0],       dtype=np.uint8)

        # Sequential: col0 establishes 2 new rows; col1, col2 zero-padded to 4.
        ref = IncrementalRREF()
        ref.add_column(H_init[:, 0], s_extra=s_init)
        ref.add_column(H_init[:, 1])
        ref.add_column(col0, s_extra=s_extra)
        ref.add_column(np.array([0, 1, 0, 0], dtype=np.uint8))   # col1 padded
        ref.add_column(np.array([1, 1, 0, 0], dtype=np.uint8))   # col2 padded

        batch = IncrementalRREFBatch()
        batch.add_columns([H_init[:, 0]], s_extra=s_init)
        batch.add_columns([H_init[:, 1]])
        batch.add_columns([col0, col1, col2], s_extra=s_extra, verify=True)

        _assert_fields_equal(batch, ref)
        assert batch.n_checks == 4
        assert batch.n_bits   == 5     # 2 init + 3 batch
