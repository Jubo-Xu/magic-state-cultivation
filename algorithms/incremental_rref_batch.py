import numpy as np
from typing import Optional

from incremental_rref import IncrementalRREF


class IncrementalRREFBatch(IncrementalRREF):
    """
    Drop-in replacement for IncrementalRREF with batch column addition.

    Extends IncrementalRREF with add_columns, which adds k columns
    simultaneously rather than one at a time via k calls to add_column.
    All existing methods (add_column, is_valid, verify) are inherited
    unchanged and produce identical results.  merge is overridden only to
    (a) return an IncrementalRREFBatch instance and (b) replace the
    sequential connecting-edge loop with a single add_columns call.

    Mathematical structure of add_columns
    --------------------------------------
    Given current state with T @ H = U (RREF) and k new column vectors:

      Step 1  Build M (n_full x k) from the k input vectors by finding the
              longest vector (length n_checks + n_new) and zero-padding all
              shorter vectors to the same length.  Zero-padding is correct
              because the caller builds every vector against the same shared
              check-index map: a shorter vector simply does not reach the
              check nodes introduced by the longer vectors, so those entries
              are genuinely zero.

              Extend T, U, H, s, s_prime, pivot_map for all n_new new check
              rows at once.  T grows as [[T_old, 0], [0, I_new]], so
              s'_new = [s'_old; s_extra] without recomputation.

      Step 2  Compute M' = T @ M (mod 2) in a single matrix multiply.
              M'[:, j] expresses column j in the current RREF row basis,
              equivalent to the k separate T @ h_j multiplies in add_column.

      Step 3  Column reduction of M'.  Columns j = 0, ..., k-1 are processed
              in order.  When column j is independent (new pivot at pivot_row),
              its row operations are propagated in-place into the remaining
              unprocessed columns M'[:, j+1:] so that each subsequent column
              sees the correct updated row basis without recomputing via T.
              T, U, and s_prime are updated immediately per row operation,
              exactly as in add_column.

              Independent case: new pivot registered; row ops propagated.
              Dependent case:   null vector recorded; nothing propagated.

              Both cases occur freely within the same pass; no pre-separation
              of the k columns is needed.

      Step 4  H extended with M; n_bits incremented by k.

    Block-diagonal optimisation
    ---------------------------
    When M'[:n_old, :] == 0 (all k columns have zeros in the pre-existing
    check rows), no column can ever yield a pivot in rows 0..n_old-1.  The
    pivot scan for each column starts directly at row n_old, skipping the
    guaranteed-zero upper block.  This fires when every fault in the batch
    touches only freshly claimed check nodes with no connection to checks
    already enclosed in the cluster.

    Relationship to add_column
    --------------------------
    Calling add_columns with a single-element list produces a result
    identical to add_column(columns[0], s_extra).  The per-column pre-padding
    of Z (done once per add_column call) is replaced by a single pre-padding
    of k zeros before the loop; null-vector indices use self.n_bits + j
    (self.n_bits not yet incremented during the loop), which for j = 0 equals
    the single-column case exactly.
    """

    def add_columns(self,
                    columns: list[np.ndarray],
                    s_extra: Optional[np.ndarray] = None,
                    verify: bool = False) -> None:
        """
        Add k columns simultaneously to H_C and update H, U, T, pivot_map,
        Z, s, s_prime, n_checks, and n_bits accordingly.

        Parameters
        ----------
        columns : list of array-like of uint8, length k
            The k new column vectors.  Each vector columns[j] has length
            n_checks + n_new_j, where n_new_j >= 0 is the number of new check
            rows introduced by column j.  All vectors are built against the
            same shared check-index map by the caller, so:
              - columns[j][i] = 1  iff fault j is connected to check i.
              - If two columns both touch new check c (at index r >= n_checks),
                both have columns[j][r] = 1 and columns[j'][r] = 1.
              - A shorter vector columns[j'] (len < max_len) is implicitly
                zero for indices beyond its length, meaning fault j' is not
                connected to those check nodes.
            M is constructed internally by zero-padding all vectors to the
            length of the longest one (max_len = n_checks + n_new).
        s_extra : array-like of uint8, shape (n_new,), or None
            Syndrome values for the n_new new check rows, in index order
            (row n_checks, n_checks+1, ..., n_checks+n_new-1).  Required
            when max_len > n_checks; must be None otherwise.
        verify : bool, default False
            If True, call verify() after the update.
        """
        if not columns:
            return

        n_cols = len(columns)
        columns = [np.asarray(c, dtype=np.uint8) for c in columns]

        # ------------------------------------------------------------------
        # Step 1: Build M by zero-padding all column vectors to the length
        # of the longest one, then extend T for all new check rows at once.
        #
        # The longest vector determines n_new: all new check rows across the
        # entire batch.  Shorter vectors are zero-padded because the caller
        # guarantees consistent indexing — a shorter vector simply does not
        # reach the check nodes that lie beyond its length, so those entries
        # are genuinely zero.  No information is lost or misaligned.
        #
        # After building M, T is extended once as block diagonal
        # [[T_old, 0], [0, I_new]] for all n_new new rows together.
        #
        # s and s_prime update:
        #   T_new @ s_new = [[T_old, 0], [0, I_new]] @ [s_old; s_extra]
        #                 = [T_old @ s_old; s_extra] = [s'_old; s_extra]
        # So we simply append s_extra to both — no recomputation.
        #
        # n_old records the row count before extension and is used in the
        # block-diagonal optimisation in Step 3.
        # ------------------------------------------------------------------
        max_len = max(len(c) for c in columns)

        assert max_len >= self.n_checks, (
            f"Longest column length {max_len} < current n_checks {self.n_checks}"
        )

        # Build M: zero-pad shorter columns to max_len
        M = np.zeros((max_len, n_cols), dtype=np.uint8)
        for j, c in enumerate(columns):
            M[:len(c), j] = c

        n_old = self.n_checks

        if max_len > self.n_checks:
            n_new = max_len - self.n_checks
            new_n = self.n_checks + n_new

            assert s_extra is not None and len(s_extra) == n_new, (
                f"s_extra of length {n_new} required for {n_new} new check rows"
            )
            s_extra = np.asarray(s_extra, dtype=np.uint8)

            # Expand T to block diagonal: [[T_old, 0], [0, I_new]]
            T_new = np.zeros((new_n, new_n), dtype=np.uint8)
            if self.n_checks > 0:
                T_new[:self.n_checks, :self.n_checks] = self.T
            T_new[self.n_checks:, self.n_checks:] = np.eye(n_new, dtype=np.uint8)
            self.T = T_new

            # Expand U: append zero rows for the new checks
            U_new = np.zeros((new_n, self.U.shape[1]), dtype=np.uint8)
            if self.n_checks > 0 and self.U.shape[1] > 0:
                U_new[:self.n_checks, :] = self.U
            self.U = U_new

            # Expand H: append zero rows (new columns not yet added)
            H_new = np.zeros((new_n, self.n_bits), dtype=np.uint8)
            if self.n_checks > 0 and self.n_bits > 0:
                H_new[:self.n_checks, :] = self.H
            self.H = H_new

            # Extend s and s_prime by appending s_extra
            self.s       = np.concatenate([self.s,       s_extra])
            self.s_prime = np.concatenate([self.s_prime, s_extra])

            self.pivot_map.extend([None] * n_new)
            self.n_checks = new_n

        else:
            assert s_extra is None, (
                "s_extra must be None when no new check rows are introduced"
            )

        # ------------------------------------------------------------------
        # Step 2: Compute M' = T @ M (mod 2).
        #
        # All k transformed columns are computed in one matrix multiply.
        # M'[:, j] = T @ M[:, j] expresses column j in the RREF row basis.
        # This replaces k separate T @ h_j vector multiplies of add_column.
        # ------------------------------------------------------------------
        M_prime = (self.T @ M) % 2

        # ------------------------------------------------------------------
        # Block-diagonal optimisation.
        #
        # If M'[:n_old, :] == 0, all k columns have zeros in the pre-existing
        # check rows.  This follows from M[:n_old, :] == 0 because T is block
        # diagonal: T_old maps the zero upper block of M to zero in M'.
        # No column can produce a pivot in rows 0..n_old-1, so the pivot scan
        # starts at n_old.  Row operations during column reduction only touch
        # rows >= n_old, leaving the zero upper block unchanged throughout.
        #
        # This fires when every fault in the batch touches only newly claimed
        # checks and has no connections to checks already in the cluster.
        # ------------------------------------------------------------------
        if n_old > 0 and np.all(M_prime[:n_old, :] == 0):
            pivot_scan_start = n_old
        else:
            pivot_scan_start = 0

        # ------------------------------------------------------------------
        # Pre-pad all existing null vectors with n_cols zeros.
        #
        # Each of the k new columns occupies one new bit index in the global
        # bit space (n_bits, n_bits+1, ..., n_bits+n_cols-1).  Existing null
        # vectors must be extended to cover these positions, all zero, since
        # the existing null space is unaffected by the new columns.
        #
        # This is done once here rather than once per column inside the loop,
        # matching the single np.append(z, 0) done inside add_column but
        # batched for all k new positions at once.
        # ------------------------------------------------------------------
        for idx in range(len(self.Z)):
            self.Z[idx] = np.concatenate(
                [self.Z[idx], np.zeros(n_cols, dtype=np.uint8)]
            )

        # ------------------------------------------------------------------
        # Step 3: Column reduction of M'.
        #
        # Process columns j = 0, ..., k-1 in order.  At the start of each
        # iteration, M'[:, j] already reflects row operations from all
        # previous independent columns, propagated in-place in earlier
        # iterations.  Both the independent and dependent cases are handled
        # inline; no pre-separation of the k columns is needed.
        # ------------------------------------------------------------------
        for j in range(n_cols):

            # Find the first unpivoted row with a 1 in column j.
            # With the block-diagonal optimisation, scanning starts at n_old.
            pivot_row = None
            for i in range(pivot_scan_start, self.n_checks):
                if M_prime[i, j] == 1 and self.pivot_map[i] is None:
                    pivot_row = i
                    break

            if pivot_row is None:
                # ----------------------------------------------------------
                # DEPENDENT CASE
                #
                # All rows where M'[i, j] = 1 are already pivoted — either
                # by pre-existing pivots or by pivots established earlier in
                # this same batch.  A null vector is constructed:
                #
                #   z[n_bits + j] = 1          — the new bit for column j
                #   z[pivot_map[i]] = 1        — for each i where M'[i,j] = 1
                #
                # pivot_map[i] may equal n_bits + j' for some j' < j when i
                # was pivoted by an earlier column in this batch; this is
                # correct because M'[:, j] has already had that column's row
                # operations propagated into it.
                #
                # No row operations are needed, so M'[:, j+1:] and T are
                # unchanged by this column.
                #
                # Null vector length is n_bits + n_cols (the final bit count
                # after all k columns), with self.n_bits not yet incremented.
                # ----------------------------------------------------------
                z = np.zeros(self.n_bits + n_cols, dtype=np.uint8)
                z[self.n_bits + j] = 1
                for i in range(self.n_checks):
                    if M_prime[i, j] == 1:
                        z[self.pivot_map[i]] = 1
                self.Z.append(z)

                # Store free variable column in U when requested
                if self.free_region_store:
                    self.U = np.hstack([self.U, M_prime[:, j].reshape(-1, 1)])

            else:
                # ----------------------------------------------------------
                # INDEPENDENT CASE
                #
                # Column j introduces a new pivot at pivot_row.  For each
                # row l != pivot_row where M'[l, j] = 1, apply the row op:
                #
                #   M'[l, j+1:] ^= M'[pivot_row, j+1:]
                #     Propagates the op into remaining unprocessed columns so
                #     that column j+1, j+2, ... each see the updated row basis
                #     without recomputing T @ h_{j'} from scratch.
                #
                #   T[l] ^= T[pivot_row]
                #     Updates T immediately so that the invariant T @ H = U
                #     is maintained and future add_column calls are correct.
                #
                #   U[l] ^= U[pivot_row]
                #     Keeps U consistent with T.  Because pivot_row had no
                #     prior pivot (pivot_map[pivot_row] was None), U[pivot_row,
                #     existing_cols] = 0 for all pre-existing pivot columns,
                #     so this op leaves those columns unchanged (XOR with 0).
                #
                #   s_prime[l] ^= s_prime[pivot_row]
                #     Mirrors the row op onto the transformed syndrome:
                #     s'_new[l] = T_new[l] @ s
                #               = (T[l] ^ T[pivot_row]) @ s
                #               = s'[l] ^ s'[pivot_row].
                # ----------------------------------------------------------
                for l in range(self.n_checks):
                    if l != pivot_row and M_prime[l, j] == 1:
                        if j + 1 < n_cols:
                            M_prime[l, j + 1:] ^= M_prime[pivot_row, j + 1:]
                        self.T[l]       ^= self.T[pivot_row]
                        self.U[l]       ^= self.U[pivot_row]
                        self.s_prime[l] ^= self.s_prime[pivot_row]

                # Append e_{pivot_row} as the new pivot column of U
                e_pivot = np.zeros(self.n_checks, dtype=np.uint8)
                e_pivot[pivot_row] = 1
                self.U = np.hstack([self.U, e_pivot.reshape(-1, 1)])

                # Register the pivot: stores the H column index (n_bits + j)
                self.pivot_map[pivot_row] = self.n_bits + j

        # ------------------------------------------------------------------
        # Step 4: Finalise H and n_bits.
        #
        # H is extended with all k new columns and n_bits incremented by k,
        # done once here rather than once per column inside the loop.
        # ------------------------------------------------------------------
        self.H = np.hstack([self.H, M])
        self.n_bits += n_cols

        if verify:
            self.verify()

    @staticmethod
    def merge(c1: 'IncrementalRREF',
              c2: 'IncrementalRREF',
              connecting_edges: list[np.ndarray],
              connecting_syndromes: Optional[list[Optional[np.ndarray]]] = None,
              verify: bool = False) -> 'IncrementalRREFBatch':
        """
        Merge two clusters c1 and c2 into a new IncrementalRREFBatch.

        Identical to IncrementalRREF.merge in every respect except:

          1. The merged object is an IncrementalRREFBatch instance, so that
             add_columns remains available for subsequent cluster growth steps
             after the merge.

          2. The sequential connecting-edge loop (one add_column per edge) is
             replaced by a single add_columns call.  The connecting_edges list
             is passed directly to add_columns, which builds M internally by
             zero-padding all vectors to the length of the longest one.  This
             is correct because all connecting edge vectors are built by the
             caller against the same unified check-index map: a shorter vector
             simply does not reach the check nodes introduced by longer vectors.

        Stage 1 — block-diagonal initialisation (identical to parent):
            T, U, H, s, s_prime, pivot_map, Z are assembled from c1 and c2
            without any column reduction.

        Stage 2 — connecting edges via add_columns:
            connecting_syndromes, if provided, must supply s_extra for the
            new check rows.  Because add_columns determines n_new from the
            longest connecting edge vector, s_extra_combined must cover all
            new check rows in the unified check-index order used by the caller.

        Parameters
        ----------
        c1, c2 : IncrementalRREF (or subclass)
            The two cluster RREFs to merge.
        connecting_edges : list of array-like of uint8
            Column vectors for faults connecting the two clusters, all built
            against the same unified check-index map.
        connecting_syndromes : list of (array-like or None), or None
            One entry per connecting edge.  Only entries corresponding to the
            edge that introduces each new check row need a non-None s_extra;
            the combined s_extra passed to add_columns is the concatenation of
            all non-None entries (covering all new check rows in index order).
        verify : bool, default False
            If True, call verify() on the merged result.

        Returns
        -------
        IncrementalRREFBatch
            The merged cluster with add_columns capability.
        """
        assert c1.free_region_store == c2.free_region_store, (
            "Cannot merge clusters with different free_region_store strategies"
        )
        # Create IncrementalRREFBatch so the result retains add_columns.
        merged = IncrementalRREFBatch(free_region_store=c1.free_region_store)

        n1_checks = c1.n_checks
        n2_checks = c2.n_checks
        n1_bits   = c1.n_bits
        n2_bits   = c2.n_bits
        merged.n_checks = n1_checks + n2_checks
        merged.n_bits   = n1_bits   + n2_bits

        # ------------------------------------------------------------------
        # Stage 1, Step 1: Block-diagonal T = [[T_C1, 0], [0, T_C2]].
        # ------------------------------------------------------------------
        merged.T = np.zeros((merged.n_checks, merged.n_checks), dtype=np.uint8)
        merged.T[:n1_checks, :n1_checks] = c1.T
        merged.T[n1_checks:, n1_checks:] = c2.T

        # ------------------------------------------------------------------
        # Stage 1, Step 2: Block-diagonal U = [[U_C1, 0], [0, U_C2]].
        #
        # c2's pivot_map entries shift by n1_bits (H column index), not by
        # u1_cols (U column index) — the two differ when free_region_store=False.
        # ------------------------------------------------------------------
        u1_cols = c1.U.shape[1]
        u2_cols = c2.U.shape[1]
        merged.U = np.zeros((merged.n_checks, u1_cols + u2_cols), dtype=np.uint8)
        merged.U[:n1_checks, :u1_cols] = c1.U
        merged.U[n1_checks:, u1_cols:] = c2.U

        # ------------------------------------------------------------------
        # Stage 1, Step 3: Block-diagonal H = [[H_C1, 0], [0, H_C2]].
        # ------------------------------------------------------------------
        merged.H = np.zeros((merged.n_checks, merged.n_bits), dtype=np.uint8)
        merged.H[:n1_checks, :n1_bits] = c1.H
        merged.H[n1_checks:, n1_bits:] = c2.H

        # ------------------------------------------------------------------
        # Stage 1, Step 4: Merge pivot_map.
        #
        # c2's pivot H-column indices shift by n1_bits (position in merged H).
        # ------------------------------------------------------------------
        pivot_map_c2_shifted = [
            (p + n1_bits if p is not None else None) for p in c2.pivot_map
        ]
        merged.pivot_map = list(c1.pivot_map) + pivot_map_c2_shifted

        # ------------------------------------------------------------------
        # Stage 1, Step 5: Merge null space bases Z.
        #
        # c1 vectors: zero-padded on the right with n2_bits zeros.
        # c2 vectors: zero-padded on the left  with n1_bits zeros.
        # ------------------------------------------------------------------
        merged.Z = []
        for z in c1.Z:
            merged.Z.append(np.concatenate([z, np.zeros(n2_bits, dtype=np.uint8)]))
        for z in c2.Z:
            merged.Z.append(np.concatenate([np.zeros(n1_bits, dtype=np.uint8), z]))

        # ------------------------------------------------------------------
        # Stage 1, Step 6: Concatenate s and s_prime.
        #
        # T_merged @ s_merged = [[T1,0],[0,T2]] @ [s1;s2]
        #                     = [T1@s1; T2@s2] = [s'1; s'2]
        # Both are simple concatenations — no recomputation needed.
        # ------------------------------------------------------------------
        merged.s       = np.concatenate([c1.s, c2.s])
        merged.s_prime = np.concatenate([c1.s_prime, c2.s_prime])

        # ------------------------------------------------------------------
        # Stage 2: Add connecting edges via add_columns.
        #
        # connecting_edges is passed directly to add_columns, which builds M
        # internally by finding the longest vector and zero-padding shorter
        # ones.  This is correct because all vectors are built against the
        # same unified check-index map by the caller.
        #
        # s_extra_combined is the concatenation of all non-None s_extra
        # entries from connecting_syndromes, in the same order as the new
        # check rows appear in the unified check-index map.
        # ------------------------------------------------------------------
        if connecting_edges:
            if connecting_syndromes is None:
                connecting_syndromes = [None] * len(connecting_edges)
            assert len(connecting_syndromes) == len(connecting_edges)

            s_extra_parts = [
                np.asarray(s, dtype=np.uint8)
                for s in connecting_syndromes
                if s is not None
            ]
            s_extra_combined = (
                np.concatenate(s_extra_parts) if s_extra_parts else None
            )
            merged.add_columns(connecting_edges, s_extra_combined)

        if verify:
            merged.verify()

        return merged
