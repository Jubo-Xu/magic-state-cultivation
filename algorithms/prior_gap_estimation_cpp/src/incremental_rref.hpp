#pragma once

#include "gf2matrix.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// IncrementalRREF
//
// Incrementally maintains the RREF decomposition of a GF(2) matrix H_C,
// along with a transformation matrix T_C (T_C @ H_C = U_C in RREF),
// a null-space basis Z_C, a pivot map, and the transformed syndrome
// s_prime = T_C @ s.
//
// All arithmetic is over GF(2): addition = XOR, multiplication = AND.
//
// Internal matrices H, T, U are stored as GF2Matrix (bitpacked, flat)
// for performance.  Z vectors are kept as byte vectors for API compatibility.
//
// pivot_map[i] = -1  means row i has no pivot (zero row in U).
// pivot_map[i] = j   means row i has its pivot at H column j.
// ---------------------------------------------------------------------------

class IncrementalRREF {
public:
    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    bool free_region_store;
    int  n_checks;   // number of rows  (checks), grows dynamically
    int  n_bits;     // number of columns (faults), grows dynamically

    // H_C: raw submatrix,              shape (n_checks, n_bits)
    // U_C: RREF of H_C,
    //   free_region_store=false: shape (n_checks, n_pivots)  [pivot cols only]
    //   free_region_store=true:  shape (n_checks, n_bits)    [all cols]
    // T_C: row-operation matrix s.t. T_C @ H_C = U_C,
    //                                  shape (n_checks, n_checks)
    GF2Matrix H, U, T;

    std::vector<int>                  pivot_map;  // -1 = no pivot
    std::vector<std::vector<uint8_t>> Z;          // null-space basis vectors

    std::vector<uint8_t> s;        // syndrome restricted to cluster checks
    std::vector<uint8_t> s_prime;  // T_C @ s  (maintained incrementally)

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    explicit IncrementalRREF(bool free_region_store = false);

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    /// Add a new fault column h (length == n_checks) — all checks existing.
    void add_column(const std::vector<uint8_t>& h);

    /// Add a new fault column h (length > n_checks) — introduces new checks.
    /// s_extra provides syndrome values for the h.size()-n_checks new rows.
    void add_column(const std::vector<uint8_t>& h,
                    const std::vector<uint8_t>& s_extra);

    /// Add k fault columns simultaneously.  Semantically equivalent to k
    /// sequential add_column calls and produces identical state (T, U, H,
    /// s_prime, pivot_map, Z).
    ///
    /// columns[j] has length n_checks + n_new_j; shorter vectors are
    /// zero-padded to max_len = n_checks + n_new internally.  All vectors
    /// must be built against the same shared check-index map by the caller.
    ///
    /// s_extra (length n_new) carries syndrome values for the new check rows
    /// introduced collectively across all columns.  Must be non-null when
    /// max_len > n_checks, null otherwise.
    ///
    /// Requires k <= 64.
    void add_columns(const std::vector<std::vector<uint8_t>>& columns,
                     const std::vector<uint8_t>* s_extra = nullptr);

    /// True iff s lies in col(H_C).  Uses s_prime — O(n_checks), no matmul.
    bool is_valid() const;

    // -----------------------------------------------------------------------
    // Merge
    // -----------------------------------------------------------------------

    static IncrementalRREF merge(
        const IncrementalRREF& c1,
        const IncrementalRREF& c2,
        const std::vector<std::vector<uint8_t>>&          connecting_edges,
        const std::vector<const std::vector<uint8_t>*>&   connecting_syndromes = {}
    );

    // -----------------------------------------------------------------------
    // Testing helpers
    // -----------------------------------------------------------------------

    /// Assert all invariants (slow — use in tests only).
    void verify() const;

private:
    void add_column_impl(const std::vector<uint8_t>& h,
                         const std::vector<uint8_t>* s_extra);

    static int gf2_rank(std::vector<std::vector<uint8_t>> M);
};
