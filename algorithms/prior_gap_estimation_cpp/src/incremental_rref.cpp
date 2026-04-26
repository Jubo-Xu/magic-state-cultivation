#include "incremental_rref.hpp"

#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

IncrementalRREF::IncrementalRREF(bool free_region_store_)
    : free_region_store(free_region_store_),
      n_checks(0),
      n_bits(0)
{}

// ---------------------------------------------------------------------------
// add_column (public overloads → shared impl)
// ---------------------------------------------------------------------------

void IncrementalRREF::add_column(const std::vector<uint8_t>& h)
{
    add_column_impl(h, nullptr);
}

void IncrementalRREF::add_column(const std::vector<uint8_t>& h,
                                  const std::vector<uint8_t>& s_extra)
{
    add_column_impl(h, &s_extra);
}

// ---------------------------------------------------------------------------
// add_column_impl
//
// Performance-critical path.  Key optimisations (all via GF2Matrix):
//   [1] Bitpacking  — h packed once into uint64_t words before any loop.
//   [2] Flat storage — T, U, H are contiguous; no pointer indirection.
//   [3] SIMD (AVX2) — row_xor() XORs 256 bits/instruction.
//   [4] Pack-h-once  — h_packed[] reused for T@h, H.append, and U.append.
// ---------------------------------------------------------------------------

void IncrementalRREF::add_column_impl(
    const std::vector<uint8_t>&  h,
    const std::vector<uint8_t>*  s_extra)
{
    assert((int)h.size() >= n_checks);

    // ------------------------------------------------------------------
    // Step 1: Extend T, U, H for any new check rows.
    //
    // T grows block-diagonally: [[T_old, 0], [0, I_new]]
    // U and H gain zero rows.
    // s and s_prime are extended by s_extra.
    // ------------------------------------------------------------------
    if ((int)h.size() > n_checks) {
        const int n_new  = (int)h.size() - n_checks;
        const int old_n  = n_checks;
        const int new_n  = old_n + n_new;

        assert(s_extra != nullptr &&
               (int)s_extra->size() == n_new);

        // T: append n_new zero columns, then n_new identity rows.
        T.append_zero_cols(n_new);       // T: old_n × new_n
        T.append_zero_rows(n_new);       // T: new_n × new_n
        for (int k = 0; k < n_new; ++k)
            T.set(old_n + k, old_n + k, 1);   // identity block

        // U and H: just zero rows.
        U.append_zero_rows(n_new);
        H.append_zero_rows(n_new);

        // Extend syndrome vectors.
        for (uint8_t v : *s_extra) {
            s.push_back(v);
            s_prime.push_back(v);
        }

        pivot_map.resize(new_n, -1);
        n_checks = new_n;
    } else {
        assert(s_extra == nullptr);
    }

    // ------------------------------------------------------------------
    // Step 2: Pack h into bitwords once.           [Opt 4]
    //
    // h_packed has ceil(n_checks/64) uint64_t words.
    // The same packed form is reused for T@h, and for appending to H and U.
    // ------------------------------------------------------------------
    const auto h_packed = GF2Matrix::pack_vec(h);

    // ------------------------------------------------------------------
    // Step 3: Compute h_prime = T @ h  (mod 2).   [Opt 1, 2, 4]
    //
    // T.matvec calls dot() on each row: parity(T.row(i) & h_packed).
    // Each word pair is reduced to one bit by __builtin_parityll.
    // ------------------------------------------------------------------
    std::vector<uint8_t> h_prime(n_checks);
    T.matvec(h_packed.data(), h_prime.data());

    // ------------------------------------------------------------------
    // Step 4: Scan h_prime for a new pivot.
    // ------------------------------------------------------------------
    int pivot_row = -1;
    for (int i = 0; i < n_checks; ++i)
        if (h_prime[i] == 1 && pivot_map[i] == -1) { pivot_row = i; break; }

    // ------------------------------------------------------------------
    // Step 5a: DEPENDENT CASE — build null vector.
    // ------------------------------------------------------------------
    if (pivot_row == -1) {
        // Pad existing null vectors with 0 for the new column index.
        for (auto& z : Z) z.push_back(0);

        // New null vector: bit n_bits (new column) = 1;
        // bit pivot_map[i] = 1 for each row i where h_prime[i] == 1.
        std::vector<uint8_t> z(n_bits + 1, 0);
        z[n_bits] = 1;
        for (int i = 0; i < n_checks; ++i)
            if (h_prime[i] == 1)
                z[pivot_map[i]] = 1;
        Z.push_back(std::move(z));

        if (free_region_store)
            U.append_col_packed(h_packed.data());   // [Opt 4]

    // ------------------------------------------------------------------
    // Step 5b: INDEPENDENT CASE — eliminate, update T, U, s_prime.
    // ------------------------------------------------------------------
    } else {
        for (int j = 0; j < n_checks; ++j) {
            if (j == pivot_row || h_prime[j] == 0) continue;
            T.row_xor(j, pivot_row);                // [Opt 2, 3] (AVX2 path)
            U.row_xor(j, pivot_row);
            s_prime[j] ^= s_prime[pivot_row];
        }

        // Append e_{pivot_row} as the new pivot column of U.
        std::vector<uint8_t> e(n_checks, 0);
        e[pivot_row] = 1;
        U.append_col(e);

        pivot_map[pivot_row] = n_bits;

        for (auto& z : Z) z.push_back(0);
    }

    // Append h as the new column of H.              [Opt 4]
    H.append_col_packed(h_packed.data());
    ++n_bits;
}

// ---------------------------------------------------------------------------
// add_columns
//
// Adds k fault columns simultaneously, equivalent to k sequential add_column
// calls but with four key optimisations over the naive loop:
//
//   [A] XOR-scan for T @ M  — reads T once instead of k times.
//   [B] Packed M rows        — one uint64_t per row (bit j = M[row][col j]);
//                              full-row XOR in the column reduction = 1 instruction.
//   [C] Single T extension   — extends T/U/H for all new rows in one shot.
//   [D] Block-diagonal skip  — skips n_old rows in every pivot scan when the
//                              upper block of M' is provably zero.
//
// See detailed explanation after the implementation.
// ---------------------------------------------------------------------------

void IncrementalRREF::add_columns(
    const std::vector<std::vector<uint8_t>>& columns,
    const std::vector<uint8_t>* s_extra)
{
    const int k = static_cast<int>(columns.size());
    if (k == 0) return;

    // Fast path: single column — zero overhead vs original add_column.
    if (k == 1) {
        add_column_impl(columns[0], s_extra);
        return;
    }

    // M_rows uses one uint64_t per row with bit j = M[row][j].
    // That representation covers up to 64 columns.
    assert(k <= 64 && "add_columns: k must be <= 64");

    // ------------------------------------------------------------------
    // Build M_rows — [Opt B]
    //
    // M_rows[r] is a uint64_t whose bit j is columns[j][r] (or 0 when
    // r >= columns[j].size(), giving implicit zero-padding).
    //
    // Iterating by column j and then by row is cache-friendly: each
    // columns[j] vector is read sequentially, and setting one bit per
    // entry into M_rows[r] touches M_rows sequentially too.
    // ------------------------------------------------------------------
    int max_len = 0;
    for (const auto& c : columns)
        max_len = std::max(max_len, static_cast<int>(c.size()));
    assert(max_len >= n_checks);

    std::vector<uint64_t> M_rows(max_len, 0);
    for (int j = 0; j < k; ++j) {
        const uint64_t jmask = 1ULL << j;
        for (int r = 0, len = static_cast<int>(columns[j].size()); r < len; ++r)
            if (columns[j][r]) M_rows[r] |= jmask;
    }

    // ------------------------------------------------------------------
    // Step 1: Extend T, U, H, s, s_prime for all n_new new rows — [Opt C]
    //
    // Sequential add_column does this n_new times (once per new check row
    // per column).  Here we do it once for all n_new rows combined,
    // saving n_new - 1 GF2Matrix reallocations and data copies.
    // ------------------------------------------------------------------
    const int n_old = n_checks;
    if (max_len > n_checks) {
        const int n_new = max_len - n_checks;
        assert(s_extra != nullptr &&
               static_cast<int>(s_extra->size()) == n_new);

        T.append_zero_cols(n_new);
        T.append_zero_rows(n_new);
        for (int ki = 0; ki < n_new; ++ki)
            T.set(n_checks + ki, n_checks + ki, 1);  // identity block

        U.append_zero_rows(n_new);
        H.append_zero_rows(n_new);

        for (uint8_t v : *s_extra) {
            s.push_back(v);
            s_prime.push_back(v);
        }
        pivot_map.resize(max_len, -1);
        n_checks = max_len;
    } else {
        assert(s_extra == nullptr);
    }

    // ------------------------------------------------------------------
    // Step 2: Compute M' = T @ M — [Opt A, the XOR-scan]
    //
    // Naive approach: k separate T.matvec calls, each reading all n²/64
    // bytes of T.  Total T reads: k × n²/64.
    //
    // XOR-scan approach:
    //   M_prime_rows[i] = XOR{ M_rows[r] : bit r of T[i] is 1 }
    //
    // For each row i of T, we iterate its set bits using __builtin_ctzll
    // (which maps to a single BSF/TZCNT instruction).  M_rows fits in ~8 KB
    // for n = 1084, staying in L1 cache throughout.  T is read exactly once
    // regardless of k.  For k ≥ 2 this is a k-fold reduction in T memory
    // traffic.
    // ------------------------------------------------------------------
    std::vector<uint64_t> M_prime_rows(n_checks, 0);
    {
        const int strd = T.stride();
        for (int i = 0; i < n_checks; ++i) {
            const uint64_t* Ti = T.row(i);
            uint64_t acc = 0;
            for (int w = 0; w < strd; ++w) {
                uint64_t word = Ti[w];
                while (word) {
                    // __builtin_ctzll: count trailing zeros = index of lowest set bit.
                    acc ^= M_rows[64 * w + __builtin_ctzll(word)];
                    word &= word - 1;   // clear lowest set bit
                }
            }
            M_prime_rows[i] = acc;
        }
    }

    // ------------------------------------------------------------------
    // Block-diagonal optimisation — [Opt D]
    //
    // If every entry of M_prime_rows[0..n_old-1] is zero, the upper block
    // of M' is entirely zero.  No column can ever produce a pivot in those
    // rows, so pivot scans start at n_old instead of 0.
    //
    // This fires when all k faults touch only freshly claimed check nodes
    // (none of the existing cluster's check rows appear in any column).
    // In that setting it saves n_old iterations per column per call.
    // ------------------------------------------------------------------
    int pivot_scan_start = 0;
    if (n_old > 0) {
        bool upper_zero = true;
        for (int i = 0; i < n_old && upper_zero; ++i)
            upper_zero = (M_prime_rows[i] == 0);
        if (upper_zero) pivot_scan_start = n_old;
    }

    // ------------------------------------------------------------------
    // Pre-pad all existing null vectors with k zeros — done once.
    //
    // Each of the k new columns occupies a new bit position (n_bits + j).
    // Existing null vectors must be extended to cover these positions.
    // Doing this once here (vs once per column inside the loop) is O(|Z|)
    // instead of O(k × |Z|).
    // ------------------------------------------------------------------
    for (auto& z : Z)
        z.resize(z.size() + k, 0);

    // ------------------------------------------------------------------
    // Step 3: Column reduction of M'
    //
    // Columns j = 0..k-1 are processed in order.  After each independent
    // column j is handled, its row operations are propagated into M' via
    // the uint64_t XOR, so column j+1 sees the correct updated basis
    // without recomputing T @ h_{j+1} from scratch.
    //
    // The full uint64_t row XOR (not just bits j+1..k-1) is correct:
    // bits 0..j-1 of M_prime_rows are for already-processed columns and
    // are never read again, so modifying them has no effect on correctness.
    // ------------------------------------------------------------------
    for (int j = 0; j < k; ++j) {
        const uint64_t col_mask = 1ULL << j;

        // Pivot scan: first unpivoted row with M'[i, j] == 1.
        int pivot_row = -1;
        for (int i = pivot_scan_start; i < n_checks; ++i) {
            if ((M_prime_rows[i] & col_mask) && pivot_map[i] == -1) {
                pivot_row = i;
                break;
            }
        }

        if (pivot_row == -1) {
            // ----------------------------------------------------------
            // DEPENDENT CASE: column j is linearly dependent on the
            // previously processed columns.  Build its null vector.
            //
            // z[n_bits + j]   = 1        (this column is a free variable)
            // z[pivot_map[i]] = 1        for each i where M'[i, j] = 1
            //
            // pivot_map[i] may point to a pre-existing pivot (< n_bits)
            // or to a pivot established by an earlier column j' < j in
            // this batch (= n_bits + j').  Both cases are correct because
            // M_prime_rows[i] already reflects the propagated row ops from
            // those earlier columns.
            //
            // The null vector is created with full length n_bits + k
            // (the final bit count after all k columns), matching the
            // k zeros pre-padded onto existing Z vectors above.
            // ----------------------------------------------------------
            std::vector<uint8_t> z(n_bits + k, 0);
            z[n_bits + j] = 1;
            for (int i = 0; i < n_checks; ++i)
                if (M_prime_rows[i] & col_mask)
                    z[pivot_map[i]] = 1;   // pivot_map[i] != -1 by invariant
            Z.push_back(std::move(z));

            if (free_region_store) {
                std::vector<uint8_t> m_prime_col_j(n_checks, 0);
                for (int i = 0; i < n_checks; ++i)
                    m_prime_col_j[i] = (M_prime_rows[i] >> j) & 1;
                auto packed = GF2Matrix::pack_vec(m_prime_col_j);
                U.append_col_packed(packed.data());
            }

        } else {
            // ----------------------------------------------------------
            // INDEPENDENT CASE: new pivot at pivot_row.
            //
            // For each row l ≠ pivot_row where M'[l, j] == 1, apply:
            //   M_prime_rows[l] ^= M_prime_rows[pivot_row]   [Opt B: 1 XOR]
            //   T[l]       ^= T[pivot_row]                   [AVX2]
            //   U[l]       ^= U[pivot_row]                   [AVX2]
            //   s_prime[l] ^= s_prime[pivot_row]
            //
            // The M_prime_rows XOR propagates this column's row operation
            // into all future columns j+1..k-1 in one 64-bit XOR, replacing
            // k-j-1 separate matvec calls in the sequential approach.
            // ----------------------------------------------------------
            const uint64_t prow_M = M_prime_rows[pivot_row];
            for (int l = 0; l < n_checks; ++l) {
                if (l == pivot_row || !(M_prime_rows[l] & col_mask)) continue;
                M_prime_rows[l] ^= prow_M;        // [Opt B] full row, 1 instruction
                T.row_xor(l, pivot_row);           // AVX2: n/64 words
                U.row_xor(l, pivot_row);           // AVX2
                s_prime[l] ^= s_prime[pivot_row];
            }

            // Append e_{pivot_row} as the new pivot column of U.
            std::vector<uint8_t> e(n_checks, 0);
            e[pivot_row] = 1;
            U.append_col(e);

            pivot_map[pivot_row] = n_bits + j;
            // Z pre-padding already covers this new position (done above).
        }
    }

    // ------------------------------------------------------------------
    // Step 4: Extend H with all k original (un-transformed) columns.
    //
    // Extract column j from M_rows (bit j of each M_rows[r]) and pack it.
    // n_bits is incremented once at the end — vs k separate increments in
    // the sequential approach, which matters for the pivot_map index
    // assignments above (n_bits + j is the correct column index for each j).
    // ------------------------------------------------------------------
    for (int j = 0; j < k; ++j) {
        std::vector<uint8_t> col_j(n_checks, 0);
        for (int r = 0; r < n_checks; ++r)
            col_j[r] = (M_rows[r] >> j) & 1;
        auto packed = GF2Matrix::pack_vec(col_j);
        H.append_col_packed(packed.data());
    }
    n_bits += k;
}

// ---------------------------------------------------------------------------
// is_valid
// ---------------------------------------------------------------------------

bool IncrementalRREF::is_valid() const
{
    for (int i = 0; i < n_checks; ++i)
        if (pivot_map[i] == -1 && s_prime[i] == 1)
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

IncrementalRREF IncrementalRREF::merge(
    const IncrementalRREF& c1,
    const IncrementalRREF& c2,
    const std::vector<std::vector<uint8_t>>&          connecting_edges,
    const std::vector<const std::vector<uint8_t>*>&   connecting_syndromes_in)
{
    assert(c1.free_region_store == c2.free_region_store);

    std::vector<const std::vector<uint8_t>*> conn_syn = connecting_syndromes_in;
    if (conn_syn.empty())
        conn_syn.assign(connecting_edges.size(), nullptr);
    assert(conn_syn.size() == connecting_edges.size());

    IncrementalRREF merged(c1.free_region_store);

    const int n1c = c1.n_checks, n2c = c2.n_checks;
    const int n1b = c1.n_bits,   n2b = c2.n_bits;
    const int nc  = n1c + n2c;
    const int nb  = n1b + n2b;

    merged.n_checks = nc;
    merged.n_bits   = nb;

    // ------------------------------------------------------------------
    // T = [[T_c1, 0], [0, T_c2]]   (block diagonal)
    // ------------------------------------------------------------------
    merged.T = GF2Matrix(nc, nc);
    merged.T.copy_block(0,   0,   c1.T, 0, 0, n1c, n1c);
    merged.T.copy_block(n1c, n1c, c2.T, 0, 0, n2c, n2c);

    // ------------------------------------------------------------------
    // U = [[U_c1, 0], [0, U_c2]]
    // ------------------------------------------------------------------
    const int u1c = c1.U.n_cols(), u2c = c2.U.n_cols();
    merged.U = GF2Matrix(nc, u1c + u2c);
    merged.U.copy_block(0,   0,   c1.U, 0, 0, n1c, u1c);
    merged.U.copy_block(n1c, u1c, c2.U, 0, 0, n2c, u2c);

    // ------------------------------------------------------------------
    // H = [[H_c1, 0], [0, H_c2]]
    // ------------------------------------------------------------------
    merged.H = GF2Matrix(nc, nb);
    merged.H.copy_block(0,   0,   c1.H, 0, 0, n1c, n1b);
    merged.H.copy_block(n1c, n1b, c2.H, 0, 0, n2c, n2b);

    // ------------------------------------------------------------------
    // pivot_map: c2's H-column indices shift by n1b.
    // ------------------------------------------------------------------
    merged.pivot_map.resize(nc, -1);
    for (int i = 0; i < n1c; ++i) merged.pivot_map[i] = c1.pivot_map[i];
    for (int i = 0; i < n2c; ++i)
        merged.pivot_map[n1c + i] =
            (c2.pivot_map[i] == -1) ? -1 : c2.pivot_map[i] + n1b;

    // ------------------------------------------------------------------
    // Z: pad c1 vectors on the right, c2 vectors on the left.
    // ------------------------------------------------------------------
    merged.Z.reserve(c1.Z.size() + c2.Z.size());
    for (const auto& z : c1.Z) {
        auto zz = z;  zz.resize(nb, 0);
        merged.Z.push_back(std::move(zz));
    }
    for (const auto& z : c2.Z) {
        std::vector<uint8_t> zz(n1b, 0);
        zz.insert(zz.end(), z.begin(), z.end());
        merged.Z.push_back(std::move(zz));
    }

    // ------------------------------------------------------------------
    // s, s_prime: concatenate.
    // ------------------------------------------------------------------
    merged.s.insert(merged.s.end(), c1.s.begin(), c1.s.end());
    merged.s.insert(merged.s.end(), c2.s.begin(), c2.s.end());
    merged.s_prime.insert(merged.s_prime.end(), c1.s_prime.begin(), c1.s_prime.end());
    merged.s_prime.insert(merged.s_prime.end(), c2.s_prime.begin(), c2.s_prime.end());

    // ------------------------------------------------------------------
    // Connecting edges — added via add_columns for one T@M pass.
    //
    // s_extra_combined is the concatenation of all non-null syndrome
    // vectors from conn_syn, in order.  This is valid because all
    // connecting-edge vectors are built against the same unified
    // check-index map: each new check row appears in at most one
    // connecting edge, so the concatenation covers them in index order.
    // ------------------------------------------------------------------
    if (!connecting_edges.empty()) {
        std::vector<uint8_t> s_extra_combined;
        for (const auto* sv : conn_syn)
            if (sv)
                s_extra_combined.insert(s_extra_combined.end(),
                                        sv->begin(), sv->end());
        const std::vector<uint8_t>* s_ptr =
            s_extra_combined.empty() ? nullptr : &s_extra_combined;
        merged.add_columns(connecting_edges, s_ptr);
    }

    return merged;
}

// ---------------------------------------------------------------------------
// verify  (tests only)
// ---------------------------------------------------------------------------

void IncrementalRREF::verify() const
{
    if (n_checks == 0) return;

    // Invariant 1: T @ s == s_prime.
    if (!s.empty()) {
        auto s_packed = GF2Matrix::pack_vec(s);
        for (int i = 0; i < n_checks; ++i) {
            uint8_t ts_i = T.dot(i, s_packed.data());
            assert(ts_i == s_prime[i] &&
                   "Invariant 1 violated: T@s != s_prime");
        }
    }

    // Invariant 2: for each pivot row i with pivot at H column j,
    //   (T @ H)[r][j] == (r == i ? 1 : 0) for all r.
    if (n_bits > 0) {
        for (int i = 0; i < n_checks; ++i) {
            if (pivot_map[i] == -1) continue;
            int j = pivot_map[i];
            // Extract column j of H as a packed vector.
            std::vector<uint8_t> hj(n_checks);
            for (int r = 0; r < n_checks; ++r) hj[r] = H.get(r, j);
            auto hj_packed = GF2Matrix::pack_vec(hj);
            for (int r = 0; r < n_checks; ++r) {
                uint8_t th = T.dot(r, hj_packed.data());
                assert(th == (r == i ? 1 : 0) &&
                       "Invariant 2 violated: pivot column of T@H not unit vector");
            }
        }
    }

    // Invariant 3: H @ z == 0 for every z in Z.
    for (size_t k = 0; k < Z.size(); ++k) {
        assert((int)Z[k].size() == n_bits);
        auto z_packed = GF2Matrix::pack_vec(Z[k]);
        for (int i = 0; i < n_checks; ++i) {
            uint8_t hz_i = H.dot(i, z_packed.data());
            assert(hz_i == 0 &&
                   "Invariant 3 violated: H@z != 0");
        }
    }

    // Invariant 4: Z vectors are linearly independent over GF(2).
    if (Z.size() > 1) {
        assert(gf2_rank(Z) == (int)Z.size() &&
               "Invariant 4 violated: Z not linearly independent");
    }

    // Invariant 5: rank-nullity — n_pivots + |Z| == n_bits.
    int n_pivots = 0;
    for (int p : pivot_map) if (p != -1) ++n_pivots;
    assert(n_pivots + (int)Z.size() == n_bits &&
           "Invariant 5 violated: rank-nullity");
}

// ---------------------------------------------------------------------------
// gf2_rank  (private static — used only by verify via Z)
// ---------------------------------------------------------------------------

int IncrementalRREF::gf2_rank(std::vector<std::vector<uint8_t>> M)
{
    if (M.empty()) return 0;
    const int rows = (int)M.size();
    const int cols = (int)M[0].size();
    int rank = 0;
    for (int col = 0; col < cols; ++col) {
        int pivot = -1;
        for (int row = rank; row < rows; ++row)
            if (M[row][col] & 1) { pivot = row; break; }
        if (pivot == -1) continue;
        std::swap(M[rank], M[pivot]);
        for (int row = 0; row < rows; ++row)
            if (row != rank && (M[row][col] & 1))
                for (int c = 0; c < cols; ++c)
                    M[row][c] ^= M[rank][c];
        ++rank;
    }
    return rank;
}
