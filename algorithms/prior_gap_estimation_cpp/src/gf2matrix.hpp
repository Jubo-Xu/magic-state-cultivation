#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

#ifdef __AVX2__
#  include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// GF2Matrix
//
// Row-major, flat bitpacked GF(2) matrix.
//
// Storage layout:
//   data_[i * stride_ + w]  =  w-th uint64_t word of row i
//   bit j of row i          =  (data_[i*stride_ + (j>>6)] >> (j&63)) & 1
//
// stride_ = ceil(n_cols_ / 64)  uint64_t words per row.
//
// Optimization map:
//   [1] Bitpacking     — 1 byte → 1 bit; inner loops shrink by 64×
//   [2] Flat storage   — single vector<uint64_t>, no pointer indirection
//   [3] SIMD (AVX2)    — row_xor XORs 256 bits per instruction (#ifdef __AVX2__)
//   [4] Pack-h-once    — caller calls pack_vec(h) once; passes pointer everywhere
// ---------------------------------------------------------------------------

class GF2Matrix {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------
    GF2Matrix() = default;
    GF2Matrix(int n_rows, int n_cols);   // zero-initialized

    int n_rows()  const { return n_rows_; }
    int n_cols()  const { return n_cols_; }
    int stride()  const { return stride_; }  // uint64_t words per row

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------
    uint8_t get(int i, int j) const
    {
        return (data_[i * stride_ + (j >> 6)] >> (j & 63)) & 1;
    }
    void set(int i, int j, uint8_t v)
    {
        uint64_t& w = data_[i * stride_ + (j >> 6)];
        uint64_t  m = 1ULL << (j & 63);
        if (v) w |= m; else w &= ~m;
    }

    // Raw pointer to row i (stride_ words).
    uint64_t*       row(int i)       { return data_.data() + i * stride_; }
    const uint64_t* row(int i) const { return data_.data() + i * stride_; }

    // ------------------------------------------------------------------
    // GF(2) row operations
    // ------------------------------------------------------------------

    /// row[dst] ^= row[src]  — uses AVX2 when compiled with -mavx2.
    void row_xor(int dst, int src);

    // ------------------------------------------------------------------
    // Packing / dot product / matvec
    // ------------------------------------------------------------------

    /// Pack a byte vector (one GF(2) value per byte) into packed uint64_t
    /// words: bit i lives at (result[i>>6] >> (i&63)) & 1.
    /// Returns ceil(h.size() / 64) words.                    [Opt 4]
    static std::vector<uint64_t> pack_vec(const std::vector<uint8_t>& h);

    /// GF(2) dot product: parity( row[i] AND h_packed[0..stride_-1] ). [Opt 1]
    /// Caller must supply at least stride_ words in h_packed.
    uint8_t dot(int i, const uint64_t* h_packed) const;

    /// result[k] = dot(k, h_packed)  for k in [0, n_rows_).
    /// result must point to at least n_rows_ bytes.
    void matvec(const uint64_t* h_packed, uint8_t* result) const;

    // ------------------------------------------------------------------
    // Structural growth  (all new entries are zero)
    // ------------------------------------------------------------------

    /// Append `count` zero rows.
    void append_zero_rows(int count);

    /// Append `count` zero columns.  Stride may increase; data preserved.
    void append_zero_cols(int count);

    /// Append one column whose bits come from h_packed:
    ///   new_col[i] = (h_packed[i>>6] >> (i&63)) & 1
    /// Caller must supply ceil(n_rows_ / 64) valid words.
    void append_col_packed(const uint64_t* h_packed);

    /// Append one column from a byte vector (length must equal n_rows_).
    void append_col(const std::vector<uint8_t>& col);

    // ------------------------------------------------------------------
    // Block copy  (used during merge; destination area must be zero)
    // ------------------------------------------------------------------

    /// Copy src[src_r0..src_r0+nrows, src_c0..src_c0+ncols) into
    ///      this[dst_r0.., dst_c0..).
    void copy_block(int dst_r0, int dst_c0,
                    const GF2Matrix& src, int src_r0, int src_c0,
                    int nrows, int ncols);

private:
    int n_rows_  = 0;
    int n_cols_  = 0;
    int stride_  = 0;          // ceil(n_cols_ / 64) words per row
    std::vector<uint64_t> data_;   // n_rows_ * stride_ words  [Opt 2]

    /// Reallocate so each row has new_stride words; zero-pad new words.
    void realign_stride(int new_stride);

    static int words_for(int n) { return (n + 63) / 64; }
};
