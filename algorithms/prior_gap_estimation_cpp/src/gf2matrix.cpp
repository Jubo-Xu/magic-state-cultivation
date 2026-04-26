#include "gf2matrix.hpp"

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GF2Matrix::GF2Matrix(int n_rows, int n_cols)
    : n_rows_(n_rows),
      n_cols_(n_cols),
      stride_(words_for(n_cols)),
      data_(n_rows * words_for(n_cols), 0)
{}

// ---------------------------------------------------------------------------
// row_xor  —  row[dst] ^= row[src]
//
// [Opt 3] With AVX2: process 4 × uint64_t = 256 bits per instruction.
// [Opt 2] No pointer-chasing: row() returns pointer into flat data_.
// ---------------------------------------------------------------------------

void GF2Matrix::row_xor(int dst, int src)
{
    uint64_t*       d = row(dst);
    const uint64_t* s = row(src);

#ifdef __AVX2__
    int w = 0;
    // Handle 4-word (256-bit) chunks.
    for (; w + 4 <= stride_; w += 4) {
        __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + w));
        __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + w));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + w),
                            _mm256_xor_si256(vd, vs));
    }
    // Scalar tail (0–3 remaining words).
    for (; w < stride_; ++w) d[w] ^= s[w];
#else
    for (int w = 0; w < stride_; ++w) d[w] ^= s[w];
#endif
}

// ---------------------------------------------------------------------------
// pack_vec
//
// [Opt 4] Pack a byte vector into bitpacked uint64_t words so that the
// packed form can be handed once to dot() and matvec() — no re-packing
// inside the loop.
// ---------------------------------------------------------------------------

std::vector<uint64_t> GF2Matrix::pack_vec(const std::vector<uint8_t>& h)
{
    const int n     = static_cast<int>(h.size());
    const int words = words_for(n);
    std::vector<uint64_t> packed(words, 0);
    for (int i = 0; i < n; ++i)
        if (h[i] & 1)
            packed[i >> 6] |= 1ULL << (i & 63);
    return packed;
}

// ---------------------------------------------------------------------------
// dot  —  GF(2) inner product of row i with a packed column vector.
//
// [Opt 1] 64 GF(2) multiplications (ANDs) and the XOR-reduction collapse
// to one __builtin_parityll call per word.
// ---------------------------------------------------------------------------

uint8_t GF2Matrix::dot(int i, const uint64_t* h) const
{
    const uint64_t* r = row(i);
    uint8_t val = 0;
    for (int w = 0; w < stride_; ++w)
        val ^= static_cast<uint8_t>(__builtin_parityll(r[w] & h[w]));
    return val;
}

// ---------------------------------------------------------------------------
// matvec  —  result[k] = dot(k, h_packed)
// ---------------------------------------------------------------------------

void GF2Matrix::matvec(const uint64_t* h_packed, uint8_t* result) const
{
    for (int i = 0; i < n_rows_; ++i)
        result[i] = dot(i, h_packed);
}

// ---------------------------------------------------------------------------
// realign_stride  —  widen each row from stride_ to new_stride words.
//
// Called at most once every 64 column additions.  The old data is
// preserved in the low words; new words are zero.
// ---------------------------------------------------------------------------

void GF2Matrix::realign_stride(int new_stride)
{
    if (new_stride == stride_) return;
    assert(new_stride > stride_);
    std::vector<uint64_t> nd(n_rows_ * new_stride, 0);
    for (int i = 0; i < n_rows_; ++i)
        std::copy(data_.data() + i * stride_,
                  data_.data() + i * stride_ + stride_,
                  nd.data()   + i * new_stride);
    data_    = std::move(nd);
    stride_  = new_stride;
}

// ---------------------------------------------------------------------------
// append_zero_rows
// ---------------------------------------------------------------------------

void GF2Matrix::append_zero_rows(int count)
{
    data_.resize((n_rows_ + count) * stride_, 0);
    n_rows_ += count;
}

// ---------------------------------------------------------------------------
// append_zero_cols
//
// The new bits are already 0 in the existing words (from construction /
// realign_stride).  Only needs to widen rows when crossing a 64-bit boundary.
// ---------------------------------------------------------------------------

void GF2Matrix::append_zero_cols(int count)
{
    const int new_stride = words_for(n_cols_ + count);
    if (new_stride != stride_) realign_stride(new_stride);
    n_cols_ += count;
}

// ---------------------------------------------------------------------------
// append_col_packed
//
// Append one column whose bit for row i is (h_packed[i>>6] >> (i&63)) & 1.
// ---------------------------------------------------------------------------

void GF2Matrix::append_col_packed(const uint64_t* h_packed)
{
    const int new_cols   = n_cols_ + 1;
    const int new_stride = words_for(new_cols);
    if (new_stride != stride_) realign_stride(new_stride);

    const int bit  = n_cols_ & 63;
    const int word = n_cols_ >> 6;
    for (int i = 0; i < n_rows_; ++i)
        if ((h_packed[i >> 6] >> (i & 63)) & 1)
            data_[i * stride_ + word] |= 1ULL << bit;

    n_cols_ = new_cols;
}

// ---------------------------------------------------------------------------
// append_col  (byte-vector form)
// ---------------------------------------------------------------------------

void GF2Matrix::append_col(const std::vector<uint8_t>& col)
{
    assert(static_cast<int>(col.size()) == n_rows_);
    const int new_cols   = n_cols_ + 1;
    const int new_stride = words_for(new_cols);
    if (new_stride != stride_) realign_stride(new_stride);

    const int bit  = n_cols_ & 63;
    const int word = n_cols_ >> 6;
    for (int i = 0; i < n_rows_; ++i)
        if (col[i] & 1)
            data_[i * stride_ + word] |= 1ULL << bit;

    n_cols_ = new_cols;
}

// ---------------------------------------------------------------------------
// copy_block
//
// Copies src[src_r0..+nrows, src_c0..+ncols) into this[dst_r0.., dst_c0..).
// Destination area is assumed zero-initialized (holds after fresh GF2Matrix
// construction), so we only need to set 1-bits.
//
// In the merge use-case src_c0 is always 0, so we optimise that path:
//
//   dst_c0 word-aligned (dst_c0 % 64 == 0):
//     Each source row is contiguous in memory → one std::copy per row.
//
//   dst_c0 bit-unaligned:
//     Each source word straddles two destination words.  We shift left/right
//     and OR into the two destination words.  A guard on the last iteration
//     prevents writing past the end of the row.
//
//   General fallback (src_c0 != 0): bit-by-bit copy.
// ---------------------------------------------------------------------------

void GF2Matrix::copy_block(int dst_r0, int dst_c0,
                            const GF2Matrix& src, int src_r0, int src_c0,
                            int nrows, int ncols)
{
    if (src_c0 == 0) {
        const int src_words = words_for(ncols);
        const int word_off  = dst_c0 >> 6;          // first dst word index
        const int dst_words = stride_ - word_off;    // words available from word_off

        if ((dst_c0 & 63) == 0) {
            // --- Word-aligned fast path -----------------------------------
            // Source row words map 1-to-1 onto destination words starting at
            // word_off.  One std::copy per row, no bit manipulation needed.
            for (int di = 0; di < nrows; ++di)
                std::copy(src.row(src_r0 + di),
                          src.row(src_r0 + di) + src_words,
                          row(dst_r0 + di) + word_off);
        } else {
            // --- Bit-unaligned path ---------------------------------------
            // Each source word contributes bits to two consecutive destination
            // words via a left-shift (lo bits) and a right-shift (hi bits).
            // The right-shift of the last source word may produce zero bits;
            // the guard (w + 1 < dst_words) ensures we never write past the
            // end of the destination row.
            const int lo = dst_c0 & 63;
            const int hi = 64 - lo;
            for (int di = 0; di < nrows; ++di) {
                uint64_t*       dp = row(dst_r0 + di) + word_off;
                const uint64_t* sp = src.row(src_r0 + di);
                for (int w = 0; w < src_words; ++w) {
                    dp[w] |= sp[w] << lo;
                    if (w + 1 < dst_words)          // guard: don't write past row end
                        dp[w + 1] |= sp[w] >> hi;
                }
            }
        }
        return;
    }

    // --- General fallback (src_c0 != 0) -----------------------------------
    // Not used in the merge path; bit-by-bit is acceptable here.
    for (int di = 0; di < nrows; ++di)
        for (int dj = 0; dj < ncols; ++dj)
            if (src.get(src_r0 + di, src_c0 + dj))
                set(dst_r0 + di, dst_c0 + dj, 1);
}
