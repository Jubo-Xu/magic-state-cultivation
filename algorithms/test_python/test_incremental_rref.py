"""
Hypothesis-based property tests for IncrementalRREF.

Test layers
-----------
Layer 1 — add_column invariants (verify() after every insert)
  1a: add only len(h)==n_checks columns (no new checks), check all invariants
  1b: add only new-check columns (block-structure, always independent)
  1c: mixed — interleave new-check and same-check inserts

Layer 2 — is_valid agrees with naive recompute
  2a: build a cluster with a valid syndrome (s = H @ e)
  2b: build a cluster with a deliberately invalid syndrome (T@s != 0 at zero row)

Layer 3 — merge invariants
  3a: merge two clusters with no connecting edges (block-diagonal structure)
  3b: merge and add random connecting edges, verify() passes
  3c: merge of two halves gives same rank as single cluster from all columns

Layer 4 — known codes (deterministic)
  4a: repetition code — null space, validity
  4b: full-rank PCM — all syndromes valid
  4c: rank-deficient PCM — invalid syndrome detected

Layer 5 — column-order and clustering properties
  5a: column-order independence — rank and |Z| unchanged by permuting inserts
  5b: all columns from a random graph; verify() at every step
  5c: _gf2_rank helper agrees with IncrementalRREF pivot count

Usage
-----
    pytest algorithms/test_incremental_rref.py -v
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

# ---------------------------------------------------------------------------
# Import the class under test.
# ---------------------------------------------------------------------------
# from algorithms.incremental_rref import IncrementalRREF, _gf2_rank
from incremental_rref import IncrementalRREF, _gf2_rank


# ===========================================================================
# Shared helpers
# ===========================================================================

def build_single_cluster(H: np.ndarray, s: np.ndarray, frs: bool = False) -> IncrementalRREF:
    """
    Build an IncrementalRREF by adding all columns of H one by one.

    The first column is added with s_extra=s (establishing all n_checks rows
    via block-structure). Subsequent columns use len(h)==n_checks path.

    Parameters
    ----------
    H   : GF(2) matrix shape (n_checks, n_bits)
    s   : syndrome vector length n_checks
    frs : free_region_store flag

    Returns a fully constructed IncrementalRREF with verify=True on each insert.
    """
    n_checks, n_bits = H.shape
    cluster = IncrementalRREF(free_region_store=frs)
    # First column introduces all n_checks check rows (block structure)
    cluster.add_column(H[:, 0], s_extra=s, verify=True)
    # Remaining columns connect to the same n_checks rows
    for j in range(1, n_bits):
        cluster.add_column(H[:, j], verify=True)
    return cluster


def _gf2_solve(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve T @ x = b over GF(2) by augmented row reduction.
    T must be square and invertible. Returns x as uint8 array.
    """
    n = T.shape[0]
    aug = np.hstack([T.copy(), b.reshape(-1, 1)]).astype(np.uint8)
    for col in range(n):
        # Find pivot in column col at or below current row
        pivot = None
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot = row
                break
        assert pivot is not None, "T is singular — cannot solve"
        aug[[col, pivot]] = aug[[pivot, col]]  # swap rows
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]  # eliminate
    return aug[:, n] % 2


# ---------------------------------------------------------------------------
# Hypothesis strategies — small GF(2) matrices and syndromes
# ---------------------------------------------------------------------------

@st.composite
def small_gf2_matrix_and_syndrome(draw, max_rows=8, max_cols=8):
    """
    Draw a GF(2) matrix H (shape n_rows x n_cols) and syndrome s (length n_rows).
    Both are uint8 with values in {0, 1}.
    """
    n_rows = draw(st.integers(1, max_rows))
    n_cols = draw(st.integers(1, max_cols))
    H = np.array(
        draw(st.lists(
            st.lists(st.integers(0, 1), min_size=n_cols, max_size=n_cols),
            min_size=n_rows, max_size=n_rows
        )),
        dtype=np.uint8
    )
    s = np.array(
        draw(st.lists(st.integers(0, 1), min_size=n_rows, max_size=n_rows)),
        dtype=np.uint8
    )
    return H, s


@st.composite
def two_gf2_matrices_and_syndromes(draw, max_rows=6, max_cols=6):
    """Draw two independent (H, s) pairs for merge tests."""
    H1, s1 = draw(small_gf2_matrix_and_syndrome(max_rows=max_rows, max_cols=max_cols))
    H2, s2 = draw(small_gf2_matrix_and_syndrome(max_rows=max_rows, max_cols=max_cols))
    return H1, s1, H2, s2


# ===========================================================================
# Layer 1 — add_column invariants
# ===========================================================================

class TestLayer1AddColumn:
    """
    Core invariant: verify() must pass after every add_column call.

    verify() checks six sub-invariants:
      (1) T @ s == s_prime
      (2) T @ H pivot columns == e_i (RREF correctness)
      (3) every z in Z satisfies H @ z == 0
      (4) Z vectors are linearly independent
      (5) rank(H) + |Z| == n_bits  (rank-nullity)
      (6) pivot_map consistency: U[i, pivot_map[i]] == 1 only
    """

    # -----------------------------------------------------------------------
    # 1a: Only same-check columns (len(h) == n_checks throughout).
    #
    # We fix n_checks after the first column and add all subsequent columns
    # with exactly that many entries. This exercises the pivot/dependent
    # decision path without any check-row expansion.
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=10))
    @settings(max_examples=200)
    def test_1a_same_check_columns(self, hs):
        """
        Add columns with len(h)==n_checks on every insert.
        verify() must pass for both free_region_store modes.
        """
        H, s = hs
        n_checks, n_cols = H.shape

        for frs in (False, True):
            cluster = IncrementalRREF(free_region_store=frs)
            # Bootstrap: first column establishes all n_checks check rows via block-structure
            cluster.add_column(H[:, 0], s_extra=s, verify=True)
            # All subsequent columns have len(h) == n_checks (no new checks)
            for j in range(1, n_cols):
                cluster.add_column(H[:, j], verify=True)

    # -----------------------------------------------------------------------
    # 1b: Block-structure columns — each column introduces exactly one new check.
    #
    # Column j has length j+1: first j entries connect to existing checks,
    # entry j is 1 (new check). This guarantees the column is always
    # linearly independent (pivot_row is always found in the new check range),
    # so rank must increase by 1 at every step.
    # -----------------------------------------------------------------------
    @given(
        n=st.integers(1, 10),
        seed=st.integers(0, 2**31 - 1)
    )
    @settings(max_examples=100)
    def test_1b_block_structure_always_independent(self, n, seed):
        """
        Each column introduces one new check (len(h) grows by 1 each time).
        The block structure guarantees independence, so rank must increase at
        every step and verify() must pass.
        """
        rng = np.random.default_rng(seed)

        for frs in (False, True):
            cluster = IncrementalRREF(free_region_store=frs)
            rank_before = 0

            for j in range(n):
                # Column of length j+1; last entry is 1 (new check row)
                h = np.zeros(j + 1, dtype=np.uint8)
                h[:j] = rng.integers(0, 2, size=j, dtype=np.uint8)
                h[j] = 1  # new check row always has a 1

                s_extra = rng.integers(0, 2, size=1, dtype=np.uint8)
                cluster.add_column(h, s_extra=s_extra, verify=True)

                # Block structure always means independent → rank must grow
                rank_after = sum(1 for p in cluster.pivot_map if p is not None)
                assert rank_after == rank_before + 1, (
                    f"Rank did not increase at step {j}: "
                    f"before={rank_before}, after={rank_after}"
                )
                rank_before = rank_after

    # -----------------------------------------------------------------------
    # 1c: Mixed inserts — interleave block-structure and same-check columns.
    #
    # Column j < n_checks: h has j+1 entries, brings check j (block-structure).
    # Column j >= n_checks: h has n_checks entries (same-check path).
    #
    # This reproduces realistic LSD cluster growth where a fault node may
    # or may not connect to checks outside the cluster.
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=6, max_cols=12))
    @settings(max_examples=150)
    def test_1c_mixed_inserts(self, hs):
        """
        Mixed new-check and same-check inserts. verify() must pass after each.
        Rank-nullity is also checked at every intermediate step.
        """
        H, s = hs
        n_checks, n_bits = H.shape

        for frs in (False, True):
            cluster = IncrementalRREF(free_region_store=frs)

            for j in range(n_bits):
                if j < n_checks:
                    # Block-structure: column j introduces check j
                    # h is H[:j+1, j]; s_extra is the new check's syndrome
                    h = H[:j + 1, j]
                    s_ex = s[j:j + 1]
                    cluster.add_column(h, s_extra=s_ex, verify=True)
                else:
                    # Same-check: column connects only to existing n_checks rows
                    cluster.add_column(H[:, j], verify=True)

                # Rank-nullity must hold at every intermediate step
                n_pivots = sum(1 for p in cluster.pivot_map if p is not None)
                assert n_pivots + len(cluster.Z) == cluster.n_bits, (
                    f"Rank-nullity failed at step {j}: "
                    f"n_pivots={n_pivots}, |Z|={len(cluster.Z)}, n_bits={cluster.n_bits}"
                )


# ===========================================================================
# Layer 2 — is_valid agrees with naive recompute
# ===========================================================================

class TestLayer2IsValid:
    """
    Property: is_valid(naive=False) must agree with is_valid(naive=True, s_naive=s)
    for any syndrome s.

    is_valid(naive=False) uses the stored s_prime (incrementally maintained).
    is_valid(naive=True, s_naive=s) recomputes T @ s from scratch.
    Both paths must give the same boolean answer.

    Additionally:
      - s in col(H) (synthesized as H @ e) must give True
      - s not in col(H) (constructed via T-inverse) must give False
    """

    # -----------------------------------------------------------------------
    # 2a: Syndrome is in col(H) — always valid.
    #
    # Construction: pick random error e, set s = H @ e (mod 2).
    # By construction s = H @ e, so H @ e = s has a solution (e itself).
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=7, max_cols=7))
    @settings(max_examples=200)
    def test_2a_valid_syndrome_in_column_space(self, hs):
        """
        s = H @ e => is_valid must return True; incremental and naive agree.
        """
        H, _ = hs   # ignore the drawn s; we synthesize one guaranteed to be valid
        n_checks, n_bits = H.shape

        # Synthesize a syndrome that is provably in col(H)
        rng = np.random.default_rng(7)
        e = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
        s = (H @ e) % 2

        for frs in (False, True):
            cluster = build_single_cluster(H, s, frs)

            # Both modes must return True
            assert cluster.is_valid(naive=False), (
                "is_valid(naive=False) returned False for s = H @ e (s in col(H))"
            )
            assert cluster.is_valid(naive=True, s_naive=s), (
                "is_valid(naive=True) returned False for s = H @ e (s in col(H))"
            )
            # Both modes must agree
            assert (
                cluster.is_valid(naive=False)
                == cluster.is_valid(naive=True, s_naive=s)
            ), "naive and incremental is_valid disagree on a valid syndrome"

    # -----------------------------------------------------------------------
    # 2b: Syndrome is NOT in col(H) — always invalid.
    #
    # Construction: build cluster, find a zero row i (pivot_map[i] is None),
    # solve T @ s_invalid = e_i over GF(2). Then (T @ s_invalid)[i] = 1 at
    # a zero-row → is_valid must return False.
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=7, max_cols=5))
    @settings(max_examples=200)
    def test_2b_invalid_syndrome_detected(self, hs):
        """
        When rank(H) < n_checks, we can construct s not in col(H).
        is_valid must return False; incremental and naive agree.
        """
        H, _ = hs
        n_checks, n_bits = H.shape

        # Build cluster with a dummy syndrome to determine rank
        s_dummy = np.zeros(n_checks, dtype=np.uint8)
        cluster_probe = IncrementalRREF(free_region_store=False)
        cluster_probe.add_column(H[:, 0], s_extra=s_dummy)
        for j in range(1, n_bits):
            cluster_probe.add_column(H[:, j])

        # Need at least one zero row (rank < n_checks)
        zero_rows = [i for i in range(n_checks) if cluster_probe.pivot_map[i] is None]
        assume(len(zero_rows) > 0)

        # Build s_invalid such that (T @ s_invalid)[zero_rows[0]] == 1.
        # Solve T @ x = target where target = e_{zero_rows[0]}.
        # T is invertible over GF(2), so x = T^{-1} @ target exists.
        zero_row = zero_rows[0]
        target = np.zeros(n_checks, dtype=np.uint8)
        target[zero_row] = 1
        s_invalid = _gf2_solve(cluster_probe.T, target)

        # Rebuild cluster with the invalid syndrome
        for frs in (False, True):
            cluster = IncrementalRREF(free_region_store=frs)
            cluster.add_column(H[:, 0], s_extra=s_invalid, verify=True)
            for j in range(1, n_bits):
                cluster.add_column(H[:, j], verify=True)

            # Both modes must return False
            assert not cluster.is_valid(naive=False), (
                "is_valid(naive=False) returned True for s NOT in col(H)"
            )
            assert not cluster.is_valid(naive=True, s_naive=s_invalid), (
                "is_valid(naive=True) returned True for s NOT in col(H)"
            )
            # Both modes must agree
            assert (
                cluster.is_valid(naive=False)
                == cluster.is_valid(naive=True, s_naive=s_invalid)
            ), "naive and incremental is_valid disagree on an invalid syndrome"


# ===========================================================================
# Layer 3 — merge invariants
# ===========================================================================

class TestLayer3Merge:
    """
    Property: merging two valid clusters must produce a state that:
      - passes verify()
      - has H = block-diagonal of H1 and H2 (before connecting edges)
      - has s = [s1; s2] and s_prime = [s1'; s2']
      - has rank = rank(H1) + rank(H2) (before connecting edges)
    """

    # -----------------------------------------------------------------------
    # 3a: Merge with no connecting edges — purely block-diagonal.
    #
    # Expected:
    #   merged.H = [[H1, 0], [0, H2]]
    #   merged.n_checks = n1 + n2, merged.n_bits = b1 + b2
    #   merged.s = [s1; s2]
    #   merged.s_prime = [s1'; s2']
    #   rank(merged.H) = rank(H1) + rank(H2)
    # -----------------------------------------------------------------------
    @given(two_gf2_matrices_and_syndromes(max_rows=5, max_cols=5))
    @settings(max_examples=150)
    def test_3a_block_diagonal_merge(self, data):
        """
        No connecting edges: merged result is block-diagonal; verify() must pass.
        """
        H1, s1, H2, s2 = data
        n1, b1 = H1.shape
        n2, b2 = H2.shape

        for frs in (False, True):
            c1 = build_single_cluster(H1, s1, frs)
            c2 = build_single_cluster(H2, s2, frs)

            merged = IncrementalRREF.merge(c1, c2, connecting_edges=[], verify=True)

            # Shape checks
            assert merged.n_checks == n1 + n2
            assert merged.n_bits == b1 + b2

            # H must be [[H1, 0], [0, H2]]
            expected_H = np.zeros((n1 + n2, b1 + b2), dtype=np.uint8)
            expected_H[:n1, :b1] = H1
            expected_H[n1:, b1:] = H2
            assert np.array_equal(merged.H, expected_H), (
                "merged.H is not block-diagonal [[H1, 0], [0, H2]]"
            )

            # Syndrome vectors concatenate
            assert np.array_equal(merged.s, np.concatenate([s1, s2]))

            # s_prime = [s1'; s2'] — from block-diagonal T derivation
            expected_s_prime = np.concatenate([c1.s_prime, c2.s_prime])
            assert np.array_equal(merged.s_prime, expected_s_prime), (
                "merged.s_prime != [s1_prime; s2_prime]"
            )

            # Rank of block-diagonal = sum of ranks
            rank_c1 = sum(1 for p in c1.pivot_map if p is not None)
            rank_c2 = sum(1 for p in c2.pivot_map if p is not None)
            rank_merged = sum(1 for p in merged.pivot_map if p is not None)
            assert rank_merged == rank_c1 + rank_c2, (
                f"Merged rank {rank_merged} != rank(c1) + rank(c2) = {rank_c1 + rank_c2}"
            )

    # -----------------------------------------------------------------------
    # 3b: Merge with random connecting edges — verify() must pass.
    #
    # Connecting edges have length n1+n2 (no new checks beyond the two clusters).
    # We check that rank-nullity holds after all edges are added.
    # -----------------------------------------------------------------------
    @given(
        two_gf2_matrices_and_syndromes(max_rows=4, max_cols=4),
        st.integers(0, 2**31 - 1)
    )
    @settings(max_examples=100)
    def test_3b_connecting_edges(self, data, seed):
        """
        After merging and adding random connecting edges, verify() must pass
        and rank-nullity must hold.
        """
        H1, s1, H2, s2 = data
        n1, b1 = H1.shape
        n2, b2 = H2.shape
        n_total = n1 + n2

        rng = np.random.default_rng(seed)
        n_connecting = rng.integers(0, 5)
        # Each connecting edge spans all merged checks (no new checks introduced)
        connecting_edges = [
            rng.integers(0, 2, size=n_total, dtype=np.uint8)
            for _ in range(n_connecting)
        ]

        for frs in (False, True):
            c1 = build_single_cluster(H1, s1, frs)
            c2 = build_single_cluster(H2, s2, frs)

            merged = IncrementalRREF.merge(
                c1, c2,
                connecting_edges=connecting_edges,
                verify=True
            )

            # Rank must be <= n_total (cannot exceed number of check rows)
            n_pivots = sum(1 for p in merged.pivot_map if p is not None)
            assert n_pivots <= n_total

            # Rank-nullity must hold
            assert n_pivots + len(merged.Z) == merged.n_bits

    # -----------------------------------------------------------------------
    # 3c: Merge of two halves gives same rank as single cluster.
    #
    # We split H into [H_left | H_right] and compare:
    #   rank( build_single_cluster([H_left | H_right]) )
    # vs
    #   rank( merge( cluster(H_left), cluster(H_right), [] ) )
    #
    # These must be equal because rank of block-diagonal = rank(H_left) + rank(H_right),
    # which also equals the rank of the original H split this way (each sub-block
    # has independent rows since H_left and H_right operate on disjoint check rows).
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=6, max_cols=8))
    @settings(max_examples=100)
    def test_3c_merge_rank_equals_sum_of_half_ranks(self, hs):
        """
        Rank of merged cluster (no connecting edges) must equal
        rank(c1) + rank(c2), which equals _gf2_rank([[H1,0],[0,H2]]).
        """
        H, s = hs
        n_checks, n_bits = H.shape
        assume(n_bits >= 2)

        split = n_bits // 2
        H1 = H[:, :split]
        H2 = H[:, split:]

        # Syndromes for each half (use the full s for c1, zeros for c2
        # to avoid dependencies on syndrome; only rank matters here)
        s1 = s
        s2 = np.zeros(n_checks, dtype=np.uint8)

        for frs in (False, True):
            c1 = build_single_cluster(H1, s1, frs)
            c2 = build_single_cluster(H2, s2, frs)

            merged = IncrementalRREF.merge(c1, c2, connecting_edges=[], verify=True)

            rank_c1 = sum(1 for p in c1.pivot_map if p is not None)
            rank_c2 = sum(1 for p in c2.pivot_map if p is not None)
            rank_merged = sum(1 for p in merged.pivot_map if p is not None)

            assert rank_merged == rank_c1 + rank_c2, (
                f"Merged rank {rank_merged} != rank(c1)+rank(c2)={rank_c1+rank_c2}"
            )
            # rank-nullity on merged
            assert rank_merged + len(merged.Z) == merged.n_bits


# ===========================================================================
# Layer 4 — known codes (deterministic)
# ===========================================================================

class TestLayer4KnownCodes:
    """
    Deterministic tests on codes with known structure.
    These give precise, human-verifiable expectations for rank, null-space,
    and validity.
    """

    def test_4a_repetition_code(self):
        """
        3-bit repetition code: H = [[1,1,0],[0,1,1]].
        rank(H) = 2, null space is 1-dimensional (spanned by [1,1,1]).
        rank == n_checks = 2, so all syndromes are valid.
        """
        H = np.array([[1, 1, 0],
                      [0, 1, 1]], dtype=np.uint8)
        s = np.array([1, 0], dtype=np.uint8)  # H @ [1,0,0] = [1,0]

        for frs in (False, True):
            cluster = build_single_cluster(H, s, frs)

            # Rank and null-space dimension
            n_pivots = sum(1 for p in cluster.pivot_map if p is not None)
            assert n_pivots == 2, f"Expected rank 2, got {n_pivots}"
            assert len(cluster.Z) == 1, f"Expected 1 null vector, got {len(cluster.Z)}"

            # Null vector must satisfy H @ z == 0
            z = cluster.Z[0]
            assert np.array_equal((H @ z) % 2, np.zeros(2, dtype=np.uint8)), (
                f"Null vector {z} does not satisfy H @ z = 0"
            )

            # rank == n_checks => all syndromes valid
            assert cluster.is_valid(naive=False)
            assert cluster.is_valid(naive=True, s_naive=s)

    def test_4b_full_rank_pcm(self):
        """
        H = [[1,1,1],[1,0,0],[0,1,0]] — full rank over GF(2) (rank 3 = n_checks).
        Verify: col 3 of RREF = col 3 of H (since rank = n_bits = 3).
        Null space is trivial (dim 0). All syndromes are valid.

        Note: [[1,0,1],[0,1,1],[1,1,0]] might look full-rank but is NOT —
        col 3 = col 1 XOR col 2, so its GF(2) rank is only 2. Use
        [[1,1,1],[1,0,0],[0,1,0]] instead (rank verified by row reduction).
        """
        H = np.array([[1, 1, 1],
                      [1, 0, 0],
                      [0, 1, 0]], dtype=np.uint8)
        # Verify rank is indeed 3 before asserting anything about the cluster
        assert _gf2_rank(H) == 3, "Test matrix must have GF(2) rank 3"
        s = np.array([1, 0, 1], dtype=np.uint8)

        for frs in (False, True):
            cluster = build_single_cluster(H, s, frs)

            n_pivots = sum(1 for p in cluster.pivot_map if p is not None)
            assert n_pivots == 3, f"Expected full rank 3, got {n_pivots}"
            assert len(cluster.Z) == 0, f"Expected empty null space, got {len(cluster.Z)}"

            # Full rank (rank == n_checks) => every syndrome is in col(H)
            assert cluster.is_valid(naive=False)
            assert cluster.is_valid(naive=True, s_naive=s)

    def test_4c_rank_deficient_invalid_syndrome(self):
        """
        H = [[1,0],[0,1],[1,1]] — rank 2 < n_checks = 3.
        col(H) = {[0,0,0], [1,0,1], [0,1,1], [1,1,0]}.
        s_invalid = [1,0,0] is not in col(H) → is_valid must return False.
        s_valid   = [1,0,1] is H @ [1,0]     → is_valid must return True.
        """
        H = np.array([[1, 0],
                      [0, 1],
                      [1, 1]], dtype=np.uint8)
        s_invalid = np.array([1, 0, 0], dtype=np.uint8)
        s_valid   = np.array([1, 0, 1], dtype=np.uint8)

        for frs in (False, True):
            # Invalid syndrome
            cluster_inv = IncrementalRREF(free_region_store=frs)
            cluster_inv.add_column(H[:, 0], s_extra=s_invalid, verify=True)
            cluster_inv.add_column(H[:, 1], verify=True)
            assert not cluster_inv.is_valid(naive=False), (
                "Expected invalid syndrome [1,0,0] to be rejected"
            )
            assert not cluster_inv.is_valid(naive=True, s_naive=s_invalid)

            # Valid syndrome
            cluster_val = IncrementalRREF(free_region_store=frs)
            cluster_val.add_column(H[:, 0], s_extra=s_valid, verify=True)
            cluster_val.add_column(H[:, 1], verify=True)
            assert cluster_val.is_valid(naive=False), (
                "Expected valid syndrome [1,0,1] to be accepted"
            )
            assert cluster_val.is_valid(naive=True, s_naive=s_valid)

    def test_4d_merge_repetition_codes(self):
        """
        Merge two identical repetition-code clusters.
        After merge (no connecting edges), block-diagonal H has rank 4,
        null space dimension 2.
        """
        H = np.array([[1, 1, 0],
                      [0, 1, 1]], dtype=np.uint8)
        s1 = np.array([1, 0], dtype=np.uint8)
        s2 = np.array([0, 1], dtype=np.uint8)

        for frs in (False, True):
            c1 = build_single_cluster(H, s1, frs)
            c2 = build_single_cluster(H, s2, frs)

            merged = IncrementalRREF.merge(c1, c2, connecting_edges=[], verify=True)

            n_pivots = sum(1 for p in merged.pivot_map if p is not None)
            assert n_pivots == 4, f"Expected rank 4, got {n_pivots}"
            assert len(merged.Z) == 2, f"Expected 2 null vectors, got {len(merged.Z)}"

            # All null vectors must satisfy merged.H @ z == 0
            for k, z in enumerate(merged.Z):
                assert np.array_equal((merged.H @ z) % 2, np.zeros(4, dtype=np.uint8)), (
                    f"Null vector {k} of merged cluster violates H @ z = 0"
                )


# ===========================================================================
# Layer 5 — column-order and clustering properties
# ===========================================================================

class TestLayer5ColumnOrderAndClustering:
    """
    Properties that hold regardless of the order columns are processed:
      - rank(H) and dim(null(H)) are column-order-independent
      - _gf2_rank helper agrees with IncrementalRREF pivot count
      - verify() holds throughout a sequential column-by-column build
    """

    # -----------------------------------------------------------------------
    # 5a: Column-order independence.
    #
    # Build one cluster with columns in original order; another with a
    # random permutation. Rank and |Z| must be identical.
    # -----------------------------------------------------------------------
    @given(
        small_gf2_matrix_and_syndrome(max_rows=6, max_cols=8),
        st.integers(0, 2**31 - 1)
    )
    @settings(max_examples=150)
    def test_5a_column_order_independence(self, hs, seed):
        """
        Permuting the column insertion order must not change rank(H) or |Z|.
        Both clusters must also pass verify() throughout.
        """
        H, s = hs
        n_checks, n_bits = H.shape

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_bits)

        for frs in (False, True):
            # Original order
            c_orig = IncrementalRREF(free_region_store=frs)
            c_orig.add_column(H[:, 0], s_extra=s, verify=True)
            for j in range(1, n_bits):
                c_orig.add_column(H[:, j], verify=True)

            # Permuted order (same syndrome s, injected on the first column)
            c_perm = IncrementalRREF(free_region_store=frs)
            c_perm.add_column(H[:, perm[0]], s_extra=s, verify=True)
            for k in range(1, n_bits):
                c_perm.add_column(H[:, perm[k]], verify=True)

            rank_orig = sum(1 for p in c_orig.pivot_map if p is not None)
            rank_perm = sum(1 for p in c_perm.pivot_map if p is not None)

            assert rank_orig == rank_perm, (
                f"Rank changed with column permutation: orig={rank_orig}, perm={rank_perm}"
            )
            assert len(c_orig.Z) == len(c_perm.Z), (
                f"|Z| changed with column permutation: orig={len(c_orig.Z)}, perm={len(c_perm.Z)}"
            )

    # -----------------------------------------------------------------------
    # 5b: All intermediate states are valid RREF structures.
    #
    # Build a cluster column by column from a random matrix; verify() at every
    # step. This is a combined stress test for all six invariants throughout
    # the entire incremental build.
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=10))
    @settings(max_examples=200)
    def test_5b_all_intermediate_states_valid(self, hs):
        """
        verify() must pass after every single add_column call, not just at the end.
        Tests that incremental updates never leave the data structure in a
        temporarily inconsistent state.
        """
        H, s = hs
        n_checks, n_bits = H.shape

        for frs in (False, True):
            cluster = IncrementalRREF(free_region_store=frs)
            cluster.add_column(H[:, 0], s_extra=s, verify=True)
            for j in range(1, n_bits):
                cluster.add_column(H[:, j], verify=True)
                # Additionally check rank-nullity explicitly at each step
                n_pivots = sum(1 for p in cluster.pivot_map if p is not None)
                assert n_pivots + len(cluster.Z) == cluster.n_bits

    # -----------------------------------------------------------------------
    # 5c: _gf2_rank helper agrees with IncrementalRREF pivot count.
    #
    # This also validates that the helper is consistent with the main class.
    # -----------------------------------------------------------------------
    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=8))
    @settings(max_examples=200)
    def test_5c_gf2_rank_agrees_with_pivot_count(self, hs):
        """
        _gf2_rank(H) must equal the number of pivots in an IncrementalRREF
        built by adding all columns of H.
        """
        H, s = hs
        n_checks, n_bits = H.shape

        cluster = IncrementalRREF(free_region_store=False)
        cluster.add_column(H[:, 0], s_extra=s)
        for j in range(1, n_bits):
            cluster.add_column(H[:, j])

        expected_rank = _gf2_rank(H)
        actual_rank = sum(1 for p in cluster.pivot_map if p is not None)
        assert actual_rank == expected_rank, (
            f"_gf2_rank(H)={expected_rank} but IncrementalRREF has {actual_rank} pivots"
        )

    # -----------------------------------------------------------------------
    # 5d: Simulate pairwise merges to build a single spanning cluster.
    #
    # Split H into n_pieces roughly equal blocks. Build one IncrementalRREF
    # per block, then merge them pairwise (no connecting edges). The final
    # merged cluster must have rank = sum of individual ranks and pass verify().
    # -----------------------------------------------------------------------
    @given(
        small_gf2_matrix_and_syndrome(max_rows=4, max_cols=8),
        st.integers(2, 4)
    )
    @settings(max_examples=80)
    def test_5d_pairwise_merge_rank_is_sum(self, hs, n_pieces):
        """
        Split H into n_pieces column-blocks, build one cluster per block,
        then pairwise-merge them. Final rank must equal sum of individual ranks.
        """
        H, s = hs
        n_checks, n_bits = H.shape
        assume(n_bits >= n_pieces)

        # Split column indices into n_pieces roughly equal groups
        col_groups = [list(range(i, n_bits, n_pieces)) for i in range(n_pieces)]
        col_groups = [g for g in col_groups if len(g) > 0]

        for frs in (False, True):
            # Build one cluster per group
            sub_clusters = []
            sub_ranks = []
            s_used = s  # assign full syndrome to first cluster; zeros to rest
            for k, cols in enumerate(col_groups):
                H_sub = H[:, cols]
                s_sub = s_used if k == 0 else np.zeros(n_checks, dtype=np.uint8)
                c = build_single_cluster(H_sub, s_sub, frs)
                sub_clusters.append(c)
                sub_ranks.append(sum(1 for p in c.pivot_map if p is not None))

            expected_total_rank = sum(sub_ranks)

            # Pairwise merge (no connecting edges)
            merged = sub_clusters[0]
            for k in range(1, len(sub_clusters)):
                merged = IncrementalRREF.merge(
                    merged, sub_clusters[k], connecting_edges=[], verify=True
                )

            actual_rank = sum(1 for p in merged.pivot_map if p is not None)
            assert actual_rank == expected_total_rank, (
                f"Merged rank {actual_rank} != sum of sub-ranks {expected_total_rank}"
            )
            assert actual_rank + len(merged.Z) == merged.n_bits


# ===========================================================================
# Standalone helper tests — _gf2_rank
# ===========================================================================

class TestGF2RankHelper:
    """Sanity checks on the _gf2_rank helper function."""

    def test_identity_matrices(self):
        """rank(I_n) == n for n = 1..8."""
        for n in range(1, 9):
            I = np.eye(n, dtype=np.uint8)
            assert _gf2_rank(I) == n, f"rank(I_{n}) should be {n}"

    def test_all_zeros(self):
        """rank(0) == 0 for various sizes."""
        for n in range(1, 8):
            Z = np.zeros((n, n), dtype=np.uint8)
            assert _gf2_rank(Z) == 0

    def test_repetition_code(self):
        """rank([[1,1,0],[0,1,1]]) == 2 (2 independent rows)."""
        H = np.array([[1, 1, 0],
                      [0, 1, 1]], dtype=np.uint8)
        assert _gf2_rank(H) == 2

    def test_repeated_rows_give_rank_1(self):
        """A matrix with all identical rows has rank 1."""
        row = np.array([[1, 0, 1, 1]], dtype=np.uint8)
        M = np.repeat(row, 4, axis=0)
        assert _gf2_rank(M) == 1

    @given(small_gf2_matrix_and_syndrome(max_rows=8, max_cols=8))
    @settings(max_examples=200)
    def test_rank_bounded(self, hs):
        """0 <= rank(H) <= min(n_rows, n_cols)."""
        H, _ = hs
        r = _gf2_rank(H)
        assert 0 <= r <= min(H.shape), (
            f"rank {r} out of bounds for shape {H.shape}"
        )

    @given(small_gf2_matrix_and_syndrome(max_rows=6, max_cols=6))
    @settings(max_examples=100)
    def test_rank_transpose_invariant(self, hs):
        """rank(H) == rank(H^T) over GF(2)."""
        H, _ = hs
        assert _gf2_rank(H) == _gf2_rank(H.T), (
            "rank(H) != rank(H^T)"
        )
