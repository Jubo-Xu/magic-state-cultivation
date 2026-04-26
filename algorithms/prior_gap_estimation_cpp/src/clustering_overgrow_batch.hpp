#pragma once

#include "incremental_rref.hpp"
#include "min_heap.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// (virtual_weight, fault_idx) heap element — same as in clustering.hpp
// but redeclared here so this header is self-contained.
using OGBHeapEntry = std::pair<double, int>;

// ---------------------------------------------------------------------------
// ClusterStateOGB
//
// All mutable state for one growing cluster.  Extends the base ClusterState
// concept with:
//   overgrow_budget : countdown for the over-grow phase.
//   L_col_packed_local : precomputed logical-observable columns for this
//                        cluster's faults, built once in
//                        _get_active_valid_clusters() and used by
//                        create_degenerate_cycle_regions().
//
// The rref field is IncrementalRREF (which now exposes add_columns()),
// so no separate IncrementalRREFBatch subclass is needed.
// ---------------------------------------------------------------------------

struct ClusterStateOGB {
    int  cluster_id;
    bool active          = true;
    bool valid           = false;
    int  overgrow_budget = -1;  // -1: not yet valid_neutral; >0: countdown; 0: done

    std::unordered_set<int> fault_nodes;
    std::unordered_set<int> check_nodes;
    std::unordered_set<int> boundary_check_nodes;
    std::unordered_set<int> enclosed_syndromes;

    // Local ↔ global index maps (ordered by insertion time, matching RREF rows/cols).
    std::vector<int>            cluster_check_idx_to_pcm_check_idx;
    std::unordered_map<int,int> pcm_check_idx_to_cluster_check_idx;
    std::vector<int>            cluster_fault_idx_to_pcm_fault_idx;

    MinHeap<OGBHeapEntry>      heap;
    std::unordered_map<int,double> dist;

    IncrementalRREF rref;   // add_columns() available since incremental_rref update

    // Precomputed per-cluster logical-observable columns.
    // L_col_packed_local[j_local] = bit i set iff L[i][cluster_fault[j_local]] == 1.
    // Length equals cluster_fault_idx_to_pcm_fault_idx.size().
    // Built once in _get_active_valid_clusters(); empty until then.
    std::vector<uint64_t> L_col_packed_local;

    explicit ClusterStateOGB(int id) : cluster_id(id) {}

    ClusterStateOGB(const ClusterStateOGB&)            = delete;
    ClusterStateOGB& operator=(const ClusterStateOGB&) = delete;
    ClusterStateOGB(ClusterStateOGB&&)                 = default;
    ClusterStateOGB& operator=(ClusterStateOGB&&)      = default;
};

// ---------------------------------------------------------------------------
// ClusterCycleInfo
//
// Classification of the null-space basis vectors of one cluster.
// flip_int encodes the observable flip pattern as a bitmask (bit k set iff
// logical observable k is flipped).  Requires n_logical <= 64.
// ---------------------------------------------------------------------------

struct ClusterCycleInfo {
    // logical-error vectors: (null-space vector z, flip_int)
    std::vector<std::pair<std::vector<uint8_t>, int64_t>> logical_errors;
    // stabilizer vectors: null-space vector z (flip_int == 0)
    std::vector<std::vector<uint8_t>> stabilizers;
};

// ---------------------------------------------------------------------------
// ClusteringOvgBatch
//
// Tanner-graph clustering with over-grow and batch fault growth.
// C++ equivalent of the Python ClusteringOvergrowBatch.
//
// Constructor inputs
// ------------------
// n_det, n_fault              : PCM dimensions.
// check_to_faults[i]          : fault indices adjacent to check i.
// fault_to_checks[j]          : check indices adjacent to fault j.
// weights[j]                  : log((1-p_j)/p_j)  (log-likelihood weight).
// n_logical                   : number of logical observables in L.
//                               Pass 0 if L is unavailable; all null vectors
//                               will be classified as stabilizers.
// L                           : logical matrix, shape (n_logical, n_fault),
//                               row-major (L[i] = row i = observable i).
//                               Consumed at construction into L_col_packed_;
//                               the raw matrix is not stored.
//
// run() parameters
// ----------------
// over_grow_step  After first reaching valid_neutral, grow for this many
//                 additional steps.  0 = original behaviour.
// bits_per_step   Fault nodes to pop per growth step.  Must be >= 1.
//                 1 = original behaviour (field-identical results).
// ---------------------------------------------------------------------------

class ClusteringOvgBatch {
public:
    ClusteringOvgBatch(
        int n_det, int n_fault,
        std::vector<std::vector<int>> check_to_faults,
        std::vector<std::vector<int>> fault_to_checks,
        std::vector<double>           weights,
        int n_logical = 0,
        std::vector<std::vector<uint8_t>> L = {}
    );

    void run(const std::vector<uint8_t>& syndrome,
             int over_grow_step = 0,
             int bits_per_step  = 1);

    // Classify each null-space basis vector as a logical error or stabilizer.
    // Must be called after run().  Requires n_logical <= 64.
    std::unordered_map<int, ClusterCycleInfo>
    create_degenerate_cycle_regions() const;

    // Convenience: run() then create_degenerate_cycle_regions() in one call.
    std::unordered_map<int, ClusterCycleInfo>
    run_and_create_degenerate_cycle_regions(
        const std::vector<uint8_t>& syndrome,
        int over_grow_step = 0,
        int bits_per_step  = 1);

    // Populated by run(): cluster_id → raw pointer into clusters_.
    // Valid until the next call to run().
    std::unordered_map<int, ClusterStateOGB*> active_valid_clusters;

    const std::vector<std::unique_ptr<ClusterStateOGB>>& clusters() const
    { return clusters_; }

private:
    // ------------------------------------------------------------------
    // Immutable PCM data
    // ------------------------------------------------------------------
    int n_det_, n_fault_;
    std::vector<std::vector<int>> check_to_faults_;
    std::vector<std::vector<int>> fault_to_checks_;
    std::vector<double>           weights_;

    // Precomputed logical-matrix columns (global, all faults).
    // L_col_packed_[j] = bit i set iff L[i][j] == 1.
    // Length n_fault_; all zeros when n_logical_ == 0.
    int n_logical_;
    std::vector<uint64_t> L_col_packed_;

    // ------------------------------------------------------------------
    // Per-run mutable state
    // ------------------------------------------------------------------
    int over_grow_step_ = 0;
    int bits_per_step_  = 1;

    std::vector<uint8_t>                         syndrome_;
    std::vector<ClusterStateOGB*>                global_check_membership_;
    std::vector<ClusterStateOGB*>                global_fault_membership_;
    std::vector<std::unique_ptr<ClusterStateOGB>> clusters_;

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------
    void clusters_initialization(const std::vector<uint8_t>& syndrome);
    void _init_each_cluster(int seed);

    void _add_fault_batch(ClusterStateOGB* cl,
                          const std::vector<std::pair<double,int>>& faults);

    ClusterStateOGB* _merge_batch(
        ClusterStateOGB* cl,
        std::unordered_set<ClusterStateOGB*>& others,
        const std::vector<int>&    connecting_js,
        const std::vector<double>& connecting_vws);

    ClusterStateOGB* _grow_one_step(ClusterStateOGB* cl);
    void             _get_active_valid_clusters();
};
