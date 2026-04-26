import numpy as np
from typing import Optional


class IncrementalRREF:
    """
    Incrementally maintains the RREF decomposition of a GF(2) matrix H_C,
    along with the transformation matrix T_C (such that T_C @ H_C = U_C),
    a null space basis Z_C, a pivot map, and the transformed syndrome s' = T @ s.

    Supports two modes of column addition:
      - New bit brings new check rows (len(h) > n_checks): block structure,
        always linearly independent.
      - New bit connects only to existing checks (len(h) == n_checks):
        independence determined by pivot_map scan on h' = T @ h.
    """

    def __init__(self, free_region_store: bool = False):
        """
        Initialize an empty IncrementalRREF with no rows or columns.
        Both n_checks and n_bits grow dynamically as checks and bits are added.

        Parameters
        ----------
        free_region_store : bool, default False
            Global strategy for whether to store free variable (dependent)
            columns in U. If True, U is fully maintained including dependent
            columns (shape n_checks x n_bits), which enables direct T @ H = U
            verification but uses more memory. If False, U stores only pivot
            columns (shape n_checks x n_pivots), sufficient for all algorithmic
            operations. This is set once at construction and used by all
            subsequent add_column, merge, and verify calls.
        """
        self.free_region_store: bool = free_region_store

        self.n_checks: int = 0  # number of rows (checks), grows dynamically
        self.n_bits: int = 0    # number of columns (bits), grows dynamically

        # H: the full submatrix H_C, shape (n_checks, n_bits)
        # stored for verification: T @ H == U (when free_region_store=True)
        self.H: np.ndarray = np.zeros((0, 0), dtype=np.uint8)

        # U_C: RREF of H_C
        # free_region_store=True:  shape (n_checks, n_bits),  all columns
        # free_region_store=False: shape (n_checks, n_pivots), pivot cols only
        self.U: np.ndarray = np.zeros((0, 0), dtype=np.uint8)

        # T_C: invertible row-operation matrix s.t. T_C @ H_C = U_C
        # shape (n_checks, n_checks), grows as new checks are added
        self.T: np.ndarray = np.zeros((0, 0), dtype=np.uint8)

        # pivot_map[i] = j means row i has its pivot at column j in U_C
        # pivot_map[i] = None means row i is a zero row (no pivot)
        self.pivot_map: list[Optional[int]] = []

        # Z_C: list of null space basis vectors, each of length n_bits
        # each z satisfies H_C @ z == 0 (mod 2)
        self.Z: list[np.ndarray] = []

        # s: the syndrome vector restricted to this cluster's checks,
        # length n_checks, grows as new checks are added
        self.s: np.ndarray = np.zeros(0, dtype=np.uint8)

        # s_prime: the transformed syndrome s' = T @ s, length n_checks.
        # Maintained incrementally so validity can be checked in O(n_checks)
        # without a matrix-vector multiply. Invariant: s_prime == T @ s always.
        self.s_prime: np.ndarray = np.zeros(0, dtype=np.uint8)

    def add_column(self,
                   h: np.ndarray,
                   s_extra: Optional[np.ndarray] = None,
                   verify: bool = False) -> None:
        """
        Add a new column (fault/bit node) to the submatrix H_C and update
        H, U, T, pivot_map, Z, s, s_prime, n_checks, and n_bits accordingly.
        Uses self.free_region_store to decide whether to maintain free variable
        columns in U.

        If len(h) > n_checks, the extra entries h[n_checks:] correspond to
        new check rows introduced alongside this column (block structure case).
        s_extra must be provided in this case, supplying the syndrome values
        for the new check rows.

        If len(h) == n_checks, the column connects only to already-enclosed
        checks. The pivot_map scan on h' = T @ h determines independence.
        s_extra must be None in this case.

        Parameters
        ----------
        h : array-like of uint8, length >= n_checks
            The new column to add. h[i] = 1 if check i is connected to this
            bit node. Entries h[n_checks:] (if any) are for new check rows.
        s_extra : array-like of uint8 or None
            Syndrome values for the new check rows introduced by this column.
            Required when len(h) > n_checks; must be None otherwise.
        verify : bool, default False
            If True, call self.verify() after the update to assert all
            invariants. Useful during testing; disable in production.
        """
        h = np.asarray(h, dtype=np.uint8)
        assert len(h) >= self.n_checks, (
            f"Column length {len(h)} is less than current n_checks {self.n_checks}"
        )

        # ------------------------------------------------------------------
        # Step 1: Extend H, T, U, s, s_prime, pivot_map for any new check rows.
        #
        # New checks enter with no prior connections to existing bits, so their
        # rows in H_C are all-zero except at the new column being added now.
        # T expands as block diagonal [[T_old, 0], [0, I_new]].
        #
        # s update: s_new = [s_old; s_extra].
        # s_prime update: T_new @ s_new = [[T_old,0],[0,I]] @ [s_old; s_extra]
        #                              = [T_old @ s_old; s_extra]
        #                              = [s_prime_old; s_extra]
        # So we simply append s_extra to both s and s_prime — no recomputation.
        # ------------------------------------------------------------------
        if len(h) > self.n_checks:
            n_new = len(h) - self.n_checks
            old_n = self.n_checks
            new_n = old_n + n_new

            assert s_extra is not None and len(s_extra) == n_new, (
                f"s_extra of length {n_new} required when adding {n_new} new checks"
            )
            s_extra = np.asarray(s_extra, dtype=np.uint8)

            # Expand T to block diagonal: [[T_old, 0], [0, I_new]]
            T_new = np.zeros((new_n, new_n), dtype=np.uint8)
            if old_n > 0:
                T_new[:old_n, :old_n] = self.T
            T_new[old_n:, old_n:] = np.eye(n_new, dtype=np.uint8)
            self.T = T_new

            # Expand U: append zero rows for the new checks
            U_new = np.zeros((new_n, self.U.shape[1]), dtype=np.uint8)
            if old_n > 0 and self.U.shape[1] > 0:
                U_new[:old_n, :] = self.U
            self.U = U_new

            # Expand H: append zero rows for the new checks (new column not yet added)
            H_new = np.zeros((new_n, self.n_bits), dtype=np.uint8)
            if old_n > 0 and self.n_bits > 0:
                H_new[:old_n, :] = self.H
            self.H = H_new

            # Extend s and s_prime by appending s_extra
            # (derivation: T_new @ s_new = [s_prime_old; I @ s_extra] = [s_prime_old; s_extra])
            self.s = np.concatenate([self.s, s_extra])
            self.s_prime = np.concatenate([self.s_prime, s_extra])

            # New check rows have no pivot yet
            self.pivot_map.extend([None] * n_new)
            self.n_checks = new_n

        else:
            assert s_extra is None, (
                "s_extra must be None when no new checks are introduced"
            )

        # ------------------------------------------------------------------
        # Step 2: Compute h' = T @ h (mod 2).
        #
        # h' expresses the new column in the RREF row basis. h'[i] = 1 means
        # check i "sees" a connection to the new bit after all row operations.
        # ------------------------------------------------------------------
        h_prime = (self.T @ h) % 2

        # ------------------------------------------------------------------
        # Step 3: Scan h' top-down for a new pivot or dependence.
        #
        # A new pivot exists at the first row i where h'[i] = 1 and
        # pivot_map[i] is None. If all such rows already have pivots, the
        # column is linearly dependent on existing columns.
        # ------------------------------------------------------------------
        pivot_row = None
        for i in range(self.n_checks):
            if h_prime[i] == 1 and self.pivot_map[i] is None:
                pivot_row = i
                break

        if pivot_row is None:
            # --------------------------------------------------------------
            # DEPENDENT CASE
            #
            # Null vector: z[new_bit] = 1, z[pivot_map[i]] = 1 for each i
            # where h'[i] = 1. T and s_prime are unchanged (no row ops needed).
            #
            # pivot_map[i] is the H column index of the pivot at row i, so
            # z[pivot_map[i]] sets the corresponding bit position in z.
            # --------------------------------------------------------------

            # Pad existing null vectors with 0 for the new bit (same as independent
            # case). This must happen in BOTH cases because n_bits always grows by 1.
            for k in range(len(self.Z)):
                self.Z[k] = np.append(self.Z[k], 0)

            z = np.zeros(self.n_bits + 1, dtype=np.uint8)
            z[self.n_bits] = 1  # new bit index (H column index of this column)
            for i in range(self.n_checks):
                if h_prime[i] == 1:
                    z[self.pivot_map[i]] = 1  # pivot_map[i] = H column index
            self.Z.append(z)

            # Only store the free variable column h' in U when requested
            if self.free_region_store:
                self.U = np.hstack([self.U, h_prime.reshape(-1, 1)])

            # s_prime is unchanged in the dependent case — no row ops on T

        else:
            # --------------------------------------------------------------
            # INDEPENDENT CASE
            #
            # For each row j != pivot_row where h'[j] = 1, apply row op:
            #   T[j] ^= T[pivot_row]              — update transform
            #   U[j] ^= U[pivot_row]              — keep U consistent with T
            #   s_prime[j] ^= s_prime[pivot_row]  — mirror row op onto s'
            #
            # s_prime update derivation: s'_new = T_new @ s, and
            # T_new[j] = T_old[j] ^ T_old[pivot_row], so
            # s'_new[j] = T_new[j] @ s = s'_old[j] ^ s'_old[pivot_row].
            # All other rows of s_prime are unchanged.
            # --------------------------------------------------------------
            for j in range(self.n_checks):
                if j != pivot_row and h_prime[j] == 1:
                    self.T[j] ^= self.T[pivot_row]
                    self.U[j] ^= self.U[pivot_row]
                    self.s_prime[j] ^= self.s_prime[pivot_row]  # mirror row op

            # Append e_{pivot_row} as the new pivot column of U
            e_pivot = np.zeros(self.n_checks, dtype=np.uint8)
            e_pivot[pivot_row] = 1
            self.U = np.hstack([self.U, e_pivot.reshape(-1, 1)])

            # Register the new pivot
            self.pivot_map[pivot_row] = self.n_bits

            # Pad existing null vectors with 0 for the new bit
            for k in range(len(self.Z)):
                self.Z[k] = np.append(self.Z[k], 0)

        # Append h as the new column of H (done for both cases)
        self.H = np.hstack([self.H, h.reshape(-1, 1)])
        self.n_bits += 1

        if verify:
            self.verify()

    def is_valid(self,
                 naive: bool = False,
                 s_naive: Optional[np.ndarray] = None) -> bool:
        """
        Check whether the stored (or provided) syndrome lies in the column
        space of H_C, i.e. whether H_C @ e = s has a solution over GF(2).

        Since T @ H_C = U and T is invertible, s is in col(H_C) iff T @ s
        is in col(U). U is in RREF, so col(U) is spanned by pivot rows only.
        Zero rows of U contribute nothing, so T @ s must be 0 at every row i
        where pivot_map[i] is None.

        Parameters
        ----------
        naive : bool, default False
            If True, recompute T @ s_naive from scratch and check validity.
            s_naive must be provided. Useful for testing against the
            incrementally maintained s_prime.
            If False, use the stored s_prime directly — O(n_checks) scan
            with no matrix multiply.
        s_naive : array-like of uint8 or None
            The full syndrome vector to check. Required when naive=True,
            must be None when naive=False.

        Returns
        -------
        bool
            True if a solution exists (cluster is valid), False otherwise.
        """
        if naive:
            assert s_naive is not None, "s_naive required when naive=True"
            s_prime_check = (self.T @ np.asarray(s_naive, dtype=np.uint8)) % 2
        else:
            assert s_naive is None, "s_naive must be None when naive=False"
            s_prime_check = self.s_prime

        for i in range(self.n_checks):
            if self.pivot_map[i] is None and s_prime_check[i] == 1:
                return False
        return True

    def verify(self) -> None:
        """
        Assert all invariants of the current state. Raises AssertionError if
        any invariant is violated. Call after each operation during testing.

        Invariants checked:
          1. T @ s == s_prime  (syndrome transform consistency)
          2. T @ H pivot columns == e_i  (pivot column correctness)
             When free_region_store=True: T @ H == U  (full RREF correctness)
             When free_region_store=False: T @ H free columns have 0 at zero rows
          3. Every z in Z satisfies H @ z == 0  (null space correctness)
          4. Z vectors are linearly independent over GF(2)
          5. rank(H) + len(Z) == n_bits  (rank-nullity theorem)
          6. pivot_map consistency: U[i, pivot_map[i]] == 1 and is the only
             1 in that column of U
        """
        if self.n_checks == 0:
            return

        TH = (self.T @ self.H) % 2 if self.n_bits > 0 else np.zeros((self.n_checks, 0), dtype=np.uint8)

        # ------------------------------------------------------------------
        # Invariant 1: T @ s == s_prime
        # ------------------------------------------------------------------
        if self.n_checks > 0 and len(self.s) > 0:
            assert np.array_equal((self.T @ self.s) % 2, self.s_prime), (
                "Invariant violated: T @ s != s_prime"
            )

        # ------------------------------------------------------------------
        # Invariant 2: T @ H correctness.
        #
        # free_region_store=True:  T @ H must equal U exactly (full RREF)
        # free_region_store=False: pivot columns of T @ H must be unit vectors;
        #   free variable columns must be zero at all zero rows of U
        #   (i.e. rows where pivot_map[i] is None)
        # ------------------------------------------------------------------
        if self.free_region_store:
            assert np.array_equal(TH, self.U), (
                "Invariant violated: T @ H != U (free_region_store=True)"
            )
        else:
            # Check pivot columns are unit vectors e_i
            for i in range(self.n_checks):
                if self.pivot_map[i] is not None:
                    j = self.pivot_map[i]
                    expected = np.zeros(self.n_checks, dtype=np.uint8)
                    expected[i] = 1
                    assert np.array_equal(TH[:, j], expected), (
                        f"Invariant violated: pivot column {j} of T @ H is not e_{i}"
                    )
            # Check free variable columns are zero at zero rows
            pivot_cols = set(p for p in self.pivot_map if p is not None)
            for j in range(self.n_bits):
                if j not in pivot_cols:
                    for i in range(self.n_checks):
                        if self.pivot_map[i] is None:
                            assert TH[i, j] == 0, (
                                f"Invariant violated: free column {j} has 1 at zero row {i} of T @ H"
                            )

        # ------------------------------------------------------------------
        # Invariant 3: every z in Z satisfies H @ z == 0
        # ------------------------------------------------------------------
        for k, z in enumerate(self.Z):
            assert np.array_equal((self.H @ z) % 2, np.zeros(self.n_checks, dtype=np.uint8)), (
                f"Invariant violated: null vector {k} does not satisfy H @ z == 0"
            )

        # ------------------------------------------------------------------
        # Invariant 4: Z vectors are linearly independent over GF(2)
        # ------------------------------------------------------------------
        if len(self.Z) > 1:
            Z_mat = np.array(self.Z, dtype=np.uint8)
            rank_Z = _gf2_rank(Z_mat)
            assert rank_Z == len(self.Z), (
                f"Invariant violated: Z vectors are not linearly independent "
                f"(rank={rank_Z}, len={len(self.Z)})"
            )

        # ------------------------------------------------------------------
        # Invariant 5: rank-nullity — rank(H) + dim(null(H)) == n_bits
        # ------------------------------------------------------------------
        n_pivots = sum(1 for p in self.pivot_map if p is not None)
        assert n_pivots + len(self.Z) == self.n_bits, (
            f"Invariant violated: rank-nullity failed "
            f"(n_pivots={n_pivots}, |Z|={len(self.Z)}, n_bits={self.n_bits})"
        )

        # ------------------------------------------------------------------
        # Invariant 6: pivot_map consistency.
        #
        # pivot_map[i] always stores the H column index of the pivot at row i.
        # U's k-th column (U[:,k]) is the pivot column for the k-th pivot in H
        # column order (i.e., pivots sorted by ascending H column index).
        #
        # When free_region_store=True: H and U have the same number of columns
        # (all columns stored), so pivot_map[i] doubles as the U column index.
        #
        # When free_region_store=False: U stores only pivot columns in insertion
        # order (= ascending H column order). We must derive the U column index
        # by sorting pivot (H column, check row) pairs.
        # ------------------------------------------------------------------
        if self.free_region_store:
            # U column index == H column index (U has all columns)
            for i in range(self.n_checks):
                if self.pivot_map[i] is not None:
                    j = self.pivot_map[i]  # same for H and U when frs=True
                    assert self.U[i, j] == 1, (
                        f"Invariant violated: U[{i}, {j}] != 1 for pivot_map[{i}]={j}"
                    )
                    for i2 in range(self.n_checks):
                        if i2 != i:
                            assert self.U[i2, j] == 0, (
                                f"Invariant violated: pivot column {j} has 1 at non-pivot row {i2}"
                            )
        else:
            # U only has pivot columns. The k-th U column corresponds to the
            # k-th pivot when pivots are sorted by H column index (insertion order).
            pivot_rows_sorted = sorted(
                (self.pivot_map[i], i)
                for i in range(self.n_checks)
                if self.pivot_map[i] is not None
            )
            for u_col, (j_H, row_i) in enumerate(pivot_rows_sorted):
                assert self.U[row_i, u_col] == 1, (
                    f"Invariant violated: U[{row_i}, {u_col}] != 1 "
                    f"for pivot at H col {j_H}"
                )
                for i2 in range(self.n_checks):
                    if i2 != row_i:
                        assert self.U[i2, u_col] == 0, (
                            f"Invariant violated: U col {u_col} (H col {j_H}) "
                            f"has 1 at non-pivot row {i2}"
                        )

    @staticmethod
    def merge(c1: "IncrementalRREF",
              c2: "IncrementalRREF",
              connecting_edges: list[np.ndarray],
              connecting_syndromes: Optional[list[Optional[np.ndarray]]] = None,
              verify: bool = False) -> "IncrementalRREF":
        """
        Merge two clusters c1 and c2 into a new IncrementalRREF representing
        their union, then incorporate each connecting fault node edge.
        Uses c1.free_region_store as the strategy for the merged cluster
        (c1 and c2 must have the same free_region_store).

        Stage 1 — block-diagonal initialization:
          T       = [[T_C1, 0], [0, T_C2]]
          U       = [[U_C1, 0], [0, U_C2]]
          H       = [[H_C1, 0], [0, H_C2]]
          s       = [s_C1; s_C2]
          s_prime = [s'_C1; s'_C2]   (derivation: T @ s = block diag result)
          pivot_map = [pivot_map_C1, pivot_map_C2 shifted by n1_bits]
          Z       = [z padded for c1, z padded for c2]

        Stage 2 — add connecting edges via add_column.

        Parameters
        ----------
        c1 : IncrementalRREF
            State of cluster 1.
        c2 : IncrementalRREF
            State of cluster 2.
        connecting_edges : list of array-like of uint8
            Each element is a column vector h of length >= (c1.n_checks + c2.n_checks),
            representing a fault node connecting the two clusters.
        connecting_syndromes : list of (array-like or None), or None
            s_extra for each connecting edge (for new checks they introduce).
            If None, all connecting edges are assumed to introduce no new checks.
        verify : bool, default False
            If True, call verify() on the merged result after all connecting
            edges are added.

        Returns
        -------
        IncrementalRREF
            A new IncrementalRREF representing the merged cluster C = C1 ∪ C2.
        """
        assert c1.free_region_store == c2.free_region_store, (
            "Cannot merge clusters with different free_region_store strategies"
        )
        merged = IncrementalRREF(free_region_store=c1.free_region_store)

        n1_checks = c1.n_checks
        n2_checks = c2.n_checks
        n1_bits = c1.n_bits
        n2_bits = c2.n_bits
        merged.n_checks = n1_checks + n2_checks
        merged.n_bits = n1_bits + n2_bits

        # ------------------------------------------------------------------
        # Step 1: Block-diagonal T = [[T_C1, 0], [0, T_C2]].
        # ------------------------------------------------------------------
        merged.T = np.zeros((merged.n_checks, merged.n_checks), dtype=np.uint8)
        merged.T[:n1_checks, :n1_checks] = c1.T
        merged.T[n1_checks:, n1_checks:] = c2.T

        # ------------------------------------------------------------------
        # Step 2: Block-diagonal U = [[U_C1, 0], [0, U_C2]].
        #
        # pivot_map indexes U columns. c2's entries shift by u1_cols (width
        # of U_C1), not by n1_bits — these differ when free_region_store=False.
        # ------------------------------------------------------------------
        u1_cols = c1.U.shape[1]
        u2_cols = c2.U.shape[1]
        merged.U = np.zeros((merged.n_checks, u1_cols + u2_cols), dtype=np.uint8)
        merged.U[:n1_checks, :u1_cols] = c1.U
        merged.U[n1_checks:, u1_cols:] = c2.U

        # ------------------------------------------------------------------
        # Step 3: Block-diagonal H = [[H_C1, 0], [0, H_C2]].
        # ------------------------------------------------------------------
        merged.H = np.zeros((merged.n_checks, merged.n_bits), dtype=np.uint8)
        merged.H[:n1_checks, :n1_bits] = c1.H
        merged.H[n1_checks:, n1_bits:] = c2.H

        # ------------------------------------------------------------------
        # Step 4: Merge pivot_map.
        #
        # pivot_map[i] stores the H column index of the pivot at check row i.
        # In the merged H = [[H1, 0], [0, H2]], c2's columns are at H positions
        # n1_bits..n1_bits+n2_bits-1, so c2's pivot H column indices shift by
        # n1_bits (NOT by u1_cols). u1_cols is only relevant for U indexing.
        # ------------------------------------------------------------------
        pivot_map_c2_shifted = [
            (p + n1_bits if p is not None else None)
            for p in c2.pivot_map
        ]
        merged.pivot_map = list(c1.pivot_map) + pivot_map_c2_shifted

        # ------------------------------------------------------------------
        # Step 5: Merge null space bases Z.
        #
        # Z entries index bit positions, so shifts use n1_bits and n2_bits.
        # ------------------------------------------------------------------
        merged.Z = []
        for z in c1.Z:
            merged.Z.append(np.concatenate([z, np.zeros(n2_bits, dtype=np.uint8)]))
        for z in c2.Z:
            merged.Z.append(np.concatenate([np.zeros(n1_bits, dtype=np.uint8), z]))

        # ------------------------------------------------------------------
        # Step 6: Concatenate s and s_prime.
        #
        # s_merged = [s_C1; s_C2]
        # s_prime_merged = T_merged @ s_merged
        #                = [[T_C1,0],[0,T_C2]] @ [s_C1; s_C2]
        #                = [T_C1 @ s_C1; T_C2 @ s_C2]
        #                = [s'_C1; s'_C2]
        # So we simply concatenate both — no recomputation needed.
        # ------------------------------------------------------------------
        merged.s = np.concatenate([c1.s, c2.s])
        merged.s_prime = np.concatenate([c1.s_prime, c2.s_prime])

        # ------------------------------------------------------------------
        # Step 7: Add each connecting edge via add_column.
        # ------------------------------------------------------------------
        if connecting_syndromes is None:
            connecting_syndromes = [None] * len(connecting_edges)
        assert len(connecting_syndromes) == len(connecting_edges)

        for h, s_ex in zip(connecting_edges, connecting_syndromes):
            merged.add_column(np.asarray(h, dtype=np.uint8), s_ex)

        if verify:
            merged.verify()

        return merged


def _gf2_rank(M: np.ndarray) -> int:
    """Compute the rank of a GF(2) matrix via row reduction."""
    M = M.copy() % 2
    rank = 0
    for col in range(M.shape[1]):
        pivot = None
        for row in range(rank, M.shape[0]):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(M.shape[0]):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
        rank += 1
    return rank
