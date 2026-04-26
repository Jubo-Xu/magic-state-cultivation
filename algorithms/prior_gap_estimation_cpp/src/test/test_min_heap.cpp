// Correctness tests for MinHeap<T>.
//
// Build (from src/):
//   g++ -std=c++17 -O2 -Wall test/test_min_heap.cpp -o test_min_heap
//   ./test_min_heap

#include "../min_heap.hpp"

#include <cassert>
#include <cstdio>
#include <utility>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Test 1: basic push / pop ordering
//
// Push values out-of-order; pop must return them in ascending order.
// ---------------------------------------------------------------------------
static void test_push_pop_order()
{
    printf("test_push_pop_order ... ");

    MinHeap<int> h;
    h.push(5);
    h.push(1);
    h.push(3);
    h.push(2);
    h.push(4);

    assert(h.size() == 5);

    int prev = -1;
    while (!h.empty()) {
        int val = h.top();
        assert(val >= prev);
        prev = val;
        h.pop();
    }
    assert(h.empty());

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 2: pair<double,int> — the actual HeapEntry type used in clustering.
//
// Min-heap on pairs: primary key = double (virtual weight),
// secondary key = int (fault index).  Matches Python heapq tuple ordering.
// ---------------------------------------------------------------------------
static void test_pair_ordering()
{
    printf("test_pair_ordering ... ");

    using HeapEntry = std::pair<double, int>;
    MinHeap<HeapEntry> h;

    h.push({3.0, 0});
    h.push({1.0, 2});
    h.push({1.0, 1});  // same weight, smaller index should come first
    h.push({2.0, 5});

    // Pop order: (1.0,1), (1.0,2), (2.0,5), (3.0,0)
    auto [w0, i0] = h.top(); h.pop();
    assert(w0 == 1.0 && i0 == 1);

    auto [w1, i1] = h.top(); h.pop();
    assert(w1 == 1.0 && i1 == 2);

    auto [w2, i2] = h.top(); h.pop();
    assert(w2 == 2.0 && i2 == 5);

    auto [w3, i3] = h.top(); h.pop();
    assert(w3 == 3.0 && i3 == 0);

    assert(h.empty());

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 3: iteration — all elements accessible without popping.
//
// Push N elements, iterate to collect them, verify the same multiset is
// present and top() still returns the minimum afterwards.
// ---------------------------------------------------------------------------
static void test_iteration()
{
    printf("test_iteration ... ");

    MinHeap<int> h;
    std::vector<int> inserted = {7, 3, 9, 1, 5};
    for (int v : inserted) h.push(v);

    // Collect via iteration.
    std::vector<int> seen(h.begin(), h.end());
    assert(seen.size() == inserted.size());

    // Same multiset (order in the vector is heap order, not sorted).
    std::sort(seen.begin(), seen.end());
    std::sort(inserted.begin(), inserted.end());
    assert(seen == inserted);

    // Heap is NOT modified by iteration: top() must still be minimum.
    assert(h.top() == 1);
    assert(h.size() == 5);

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 4: absorb — bulk merge of two heaps.
//
// Mirrors the Python cluster-merge pattern:
//   for entry in other.heap:
//       heapq.heappush(larger.heap, entry)
//
// Verify:
//   (a) All elements from both heaps appear in the merged heap.
//   (b) Pop order is globally sorted (min-heap property preserved).
//   (c) The source heap is cleared after absorb.
// ---------------------------------------------------------------------------
static void test_absorb()
{
    printf("test_absorb ... ");

    MinHeap<int> larger, smaller;

    // Push into larger.
    for (int v : {10, 2, 8, 4}) larger.push(v);
    // Push into smaller.
    for (int v : {7, 1, 5, 3}) smaller.push(v);

    larger.absorb(smaller);

    // Source heap must be cleared.
    assert(smaller.empty());

    // Merged heap must have all 8 elements.
    assert(larger.size() == 8);

    // Pop order must be non-decreasing (global min-heap property).
    std::vector<int> expected = {1, 2, 3, 4, 5, 7, 8, 10};
    std::vector<int> got;
    while (!larger.empty()) {
        got.push_back(larger.top());
        larger.pop();
    }
    assert(got == expected);

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 5: absorb with empty operands.
//
// absorb(empty)   → no change to dst, src stays empty.
// empty.absorb(h) → dst gains all elements, src cleared.
// ---------------------------------------------------------------------------
static void test_absorb_empty()
{
    printf("test_absorb_empty ... ");

    // Case A: absorb an empty heap.
    MinHeap<int> h;
    for (int v : {3, 1, 2}) h.push(v);
    MinHeap<int> empty;
    h.absorb(empty);
    assert(h.size() == 3);
    assert(empty.empty());
    assert(h.top() == 1);

    // Case B: empty heap absorbs a non-empty one.
    MinHeap<int> dst;
    MinHeap<int> src;
    for (int v : {5, 2, 4}) src.push(v);
    dst.absorb(src);
    assert(src.empty());
    assert(dst.size() == 3);
    assert(dst.top() == 2);

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// Test 6: stale-entry / lazy-deletion pattern (mirrors clustering usage).
//
// The same fault index may appear multiple times with different virtual
// weights.  The caller discards stale entries by checking against a dist map.
// Verify that the heap correctly surfaces the smallest weight first so the
// skip logic works as expected.
// ---------------------------------------------------------------------------
static void test_lazy_deletion_pattern()
{
    printf("test_lazy_deletion_pattern ... ");

    using HeapEntry = std::pair<double, int>;
    MinHeap<HeapEntry> h;

    // Push fault 7 twice: first with weight 5.0, then updated to 2.0.
    h.push({5.0, 7});
    h.push({2.0, 7});
    h.push({3.0, 3});

    // dist map stores the best (lowest) weight seen per fault.
    std::unordered_map<int,double> dist = {{7, 2.0}, {3, 3.0}};

    // Pop loop that skips stale entries — same logic as _grow_one_step.
    std::vector<std::pair<double,int>> accepted;
    while (!h.empty()) {
        auto [vw, j] = h.top(); h.pop();
        auto it = dist.find(j);
        if (it == dist.end() || vw > it->second) continue;  // stale
        accepted.push_back({vw, j});
    }

    // Must have accepted (2.0,7) and (3.0,3), skipped stale (5.0,7).
    assert(accepted.size() == 2);
    assert(accepted[0] == std::make_pair(2.0, 7));
    assert(accepted[1] == std::make_pair(3.0, 3));

    printf("PASSED\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    test_push_pop_order();
    test_pair_ordering();
    test_iteration();
    test_absorb();
    test_absorb_empty();
    test_lazy_deletion_pattern();

    printf("\nAll tests passed.\n");
    return 0;
}
