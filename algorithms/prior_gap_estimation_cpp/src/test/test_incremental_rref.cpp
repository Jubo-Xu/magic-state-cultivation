// Standalone correctness tests for IncrementalRREF.
// Mirrors the test logic in test_clustering.py (the RREF-specific parts).
// Build (from src/):
//   g++ -std=c++17 -O2 -mavx2 test/test_incremental_rref.cpp
//       incremental_rref.cpp gf2matrix.cpp -o test_rref && ./test_rref

#include "../incremental_rref.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check_null_vectors(const IncrementalRREF& rref)
{
    // Every z in Z must satisfy H @ z == 0 mod 2.
    for (const auto& z : rref.Z) {
        assert((int)z.size() == rref.n_bits);
        for (int i = 0; i < rref.n_checks; ++i) {
            uint8_t hz = 0;
            for (int j = 0; j < rref.n_bits; ++j)
                hz ^= rref.H.get(i, j) & z[j];   // .get() replaces [i][j]
            assert(hz == 0);
        }
    }
}

static void check_rank_nullity(const IncrementalRREF& rref)
{
    int n_pivots = 0;
    for (int p : rref.pivot_map) if (p != -1) ++n_pivots;
    assert(n_pivots + (int)rref.Z.size() == rref.n_bits);
}

// ---------------------------------------------------------------------------
// Test 1: repetition code (chain graph), n=4
// ---------------------------------------------------------------------------
static void test_rep_code()
{
    printf("test_rep_code ... ");

    IncrementalRREF rref;

    rref.add_column({1}, {0});
    assert(rref.n_checks == 1);
    assert(rref.n_bits   == 1);
    assert(rref.Z.empty());
    rref.verify();

    rref.add_column({1, 1}, {0});
    assert(rref.n_checks == 2);
    assert(rref.n_bits   == 2);
    assert(rref.Z.empty());
    rref.verify();

    rref.add_column({0, 1, 1}, {0});
    assert(rref.n_checks == 3);
    assert(rref.n_bits   == 3);
    assert(rref.Z.empty());
    rref.verify();

    rref.add_column({0, 0, 1});
    assert(rref.n_checks == 3);
    assert(rref.n_bits   == 4);
    assert(rref.Z.size() == 1);

    check_null_vectors(rref);
    check_rank_nullity(rref);
    rref.verify();

    // Manually set syndrome [1,0,1] and check is_valid.
    rref.s       = {1, 0, 1};
    rref.s_prime = rref.s;
    // Recompute s_prime = T @ s using GF2Matrix.matvec.
    {
        auto sp = GF2Matrix::pack_vec(rref.s);
        rref.T.matvec(sp.data(), rref.s_prime.data());
    }
    assert(rref.is_valid());

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 2: ring code (cycle graph), n=3
// ---------------------------------------------------------------------------
static void test_ring_code()
{
    printf("test_ring_code ... ");

    IncrementalRREF rref;

    rref.add_column({1, 1}, {0, 0});
    rref.verify();

    rref.add_column({0, 1, 1}, {0});
    rref.verify();

    rref.add_column({1, 0, 1});
    assert(rref.Z.size() == 1);

    check_null_vectors(rref);
    check_rank_nullity(rref);
    rref.verify();

    // All-ones syndrome: invalid for a cycle (sum = 3, odd).
    {
        std::vector<uint8_t> s_ones(3, 1);
        auto sp = GF2Matrix::pack_vec(s_ones);
        rref.T.matvec(sp.data(), rref.s_prime.data());
        rref.s = s_ones;
    }
    assert(!rref.is_valid());

    // Zero syndrome: always valid.
    rref.s       = {0, 0, 0};
    rref.s_prime = {0, 0, 0};
    assert(rref.is_valid());

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 3: merge of two rep-2 clusters
// ---------------------------------------------------------------------------
static void test_merge()
{
    printf("test_merge ... ");

    IncrementalRREF c1;
    c1.add_column({1, 1}, {0, 0});
    c1.verify();

    IncrementalRREF c2;
    c2.add_column({1, 1}, {0, 0});
    c2.verify();

    // Connecting fault: check 1 (c1) and check 2 (c2 offset by 2).
    std::vector<std::vector<uint8_t>> edges = {{0, 1, 1, 0}};
    IncrementalRREF merged = IncrementalRREF::merge(c1, c2, edges);

    assert(merged.n_checks == 4);
    assert(merged.n_bits   == 3);

    check_null_vectors(merged);
    check_rank_nullity(merged);
    merged.verify();

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 4: dependent column produces correct null vector
// ---------------------------------------------------------------------------
static void test_dependent_column()
{
    printf("test_dependent_column ... ");

    IncrementalRREF rref;
    rref.add_column({1}, {0});          // fault 0: check 0
    rref.add_column({0, 1}, {0});       // fault 1: check 1 (new)
    rref.add_column({1, 1});            // fault 2: fault0 XOR fault1 → dependent

    assert(rref.Z.size() == 1);
    const auto& z = rref.Z[0];
    assert((int)z.size() == 3);
    for (int i = 0; i < rref.n_checks; ++i) {
        uint8_t hz = 0;
        for (int j = 0; j < rref.n_bits; ++j)
            hz ^= rref.H.get(i, j) & z[j];
        assert(hz == 0);
    }
    rref.verify();

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    test_rep_code();
    test_ring_code();
    test_merge();
    test_dependent_column();
    printf("\nAll tests passed.\n");
    return 0;
}
