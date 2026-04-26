#pragma once

// ---------------------------------------------------------------------------
// MinHeap<T, Compare>
//
// A min-heap backed by a plain std::vector, mirroring Python's heapq module.
//
// Because the underlying vector is exposed via begin()/end(), elements can
// be iterated without popping — exactly like iterating a Python heapq list.
// This enables the O(|src| + |dst|) absorb() operation used during cluster
// merges instead of the O(|src| log |dst|) pop-and-push loop that
// std::priority_queue would require.
//
// Template parameters
// -------------------
// T       : element type.  Must support Compare.
// Compare : strict-weak ordering.  Defaults to std::greater<T>, which turns
//           the underlying make_heap / push_heap / pop_heap max-heap into a
//           min-heap (smallest element at top).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <functional>
#include <vector>

template<typename T, typename Compare = std::greater<T>>
class MinHeap {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    MinHeap() = default;
    explicit MinHeap(Compare cmp) : cmp_(cmp) {}

    // -----------------------------------------------------------------------
    // Core heap operations
    // -----------------------------------------------------------------------

    /// Insert val and restore the heap property.  O(log n).
    void push(const T& val) {
        data_.push_back(val);
        std::push_heap(data_.begin(), data_.end(), cmp_);
    }
    void push(T&& val) {
        data_.push_back(std::move(val));
        std::push_heap(data_.begin(), data_.end(), cmp_);
    }

    /// Return the minimum element.  Undefined if empty.  O(1).
    const T& top() const { return data_.front(); }

    /// Remove the minimum element.  Undefined if empty.  O(log n).
    void pop() {
        std::pop_heap(data_.begin(), data_.end(), cmp_);
        data_.pop_back();
    }

    bool   empty() const { return data_.empty(); }
    size_t size()  const { return data_.size();  }
    void   clear()       { data_.clear(); }

    // -----------------------------------------------------------------------
    // Iteration  (like "for entry in heap:" in Python)
    //
    // Elements are in heap order (not sorted), exactly as Python's heapq list.
    // -----------------------------------------------------------------------
    using iterator       = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator       begin()        { return data_.begin(); }
    iterator       end()          { return data_.end();   }
    const_iterator begin()  const { return data_.begin(); }
    const_iterator end()    const { return data_.end();   }
    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend()   const { return data_.cend();   }

    // -----------------------------------------------------------------------
    // absorb(src)
    //
    // Bulk-merge all elements of src into this heap, then clear src.
    //
    // Algorithm
    // ---------
    //   1. Append all src elements to data_ in O(|src|).
    //   2. Call std::make_heap once in O(|this| + |src|).
    //   3. Clear src.
    //
    // Total: O(|this| + |src|)  vs  O(|src| log |this|) for a pop-loop.
    // This matches the Python pattern:
    //   for entry in other.heap:
    //       heapq.heappush(larger.heap, entry)
    // which is O(|src| log |this|), so absorb() is asymptotically better.
    // -----------------------------------------------------------------------
    void absorb(MinHeap& src) {
        data_.insert(data_.end(), src.data_.begin(), src.data_.end());
        std::make_heap(data_.begin(), data_.end(), cmp_);
        src.data_.clear();
    }

private:
    std::vector<T> data_;
    Compare        cmp_{};
};
