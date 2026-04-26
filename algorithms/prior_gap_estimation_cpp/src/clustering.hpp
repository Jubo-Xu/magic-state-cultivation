#pragma once

#include "incremental_rref.hpp"
#include "min_heap.hpp"

#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// ClusterState
//
// All mutable state for one growing cluster.  Non-copyable (RREF is large);
// move-only.  Raw pointers to ClusterState objects are stable as long as
// Clustering keeps them in vector<unique_ptr<ClusterState>>.
// ---------------------------------------------------------------------------

// (virtual_weight, fault_idx) — element type for all Dijkstra heaps.
using HeapEntry = std::pair<double, int>;

struct ClusterState {
    int  cluster_id;
    bool active = true;
    bool valid  = false;

    // Global PCM index sets.
    std::unordered_set<int> fault_nodes;
    std::unordered_set<int> check_nodes;
    std::unordered_set<int> boundary_check_nodes;
    std::unordered_set<int> enclosed_syndromes;

    // Local ↔ global index maps (ordered by insertion time, matching RREF rows/cols).
    std::vector<int>            cluster_check_idx_to_pcm_check_idx;
    std::unordered_map<int,int> pcm_check_idx_to_cluster_check_idx;
    std::vector<int>            cluster_fault_idx_to_pcm_fault_idx;

    // Dijkstra min-heap: (virtual_weight, fault_idx).
    MinHeap<HeapEntry> heap;
    std::unordered_map<int,double> dist;   // fault_idx → best vw seen

    IncrementalRREF rref;

    explicit ClusterState(int id) : cluster_id(id) {}

    ClusterState(const ClusterState&)            = delete;
    ClusterState& operator=(const ClusterState&) = delete;
    ClusterState(ClusterState&&)                 = default;
    ClusterState& operator=(ClusterState&&)      = default;
};

// ---------------------------------------------------------------------------
// Clustering
//
// Tanner-graph clustering for gap estimation.  Mirrors clustering.py exactly.
//
// Construct once from pre-computed adjacency data; call run(syndrome) per shot.
//
// Parameters passed to constructor
// ---------------------------------
// n_det              : number of detector / check nodes
// n_fault            : number of fault nodes
// check_to_faults[i] : fault indices adjacent to check i
// fault_to_checks[j] : check indices adjacent to fault j
// weights[j]         : log((1-p_j)/p_j)  (log-likelihood ratio; matches Python _build_adjacency)
// ---------------------------------------------------------------------------

class Clustering {
public:
    Clustering(int n_det, int n_fault,
               std::vector<std::vector<int>> check_to_faults,
               std::vector<std::vector<int>> fault_to_checks,
               std::vector<double>           weights);

    // Run the full clustering algorithm for one syndrome vector (length n_det).
    void run(const std::vector<uint8_t>& syndrome);

    // Populated by run(): cluster_id → pointer into clusters()
    std::unordered_map<int, ClusterState*> active_valid_clusters;

    // All clusters (active and inactive), in seed-detector order.
    const std::vector<std::unique_ptr<ClusterState>>& clusters() const
    { return clusters_; }

private:
    // Immutable PCM data
    int n_det_, n_fault_;
    std::vector<std::vector<int>> check_to_faults_;
    std::vector<std::vector<int>> fault_to_checks_;
    std::vector<double>           weights_;

    // Per-run state (reset at the start of each run())
    std::vector<uint8_t>                         syndrome_;
    std::vector<ClusterState*>                   global_check_membership_;  // null = unclaimed
    std::vector<ClusterState*>                   global_fault_membership_;
    std::vector<std::unique_ptr<ClusterState>>   clusters_;

    void          clusters_initialization(const std::vector<uint8_t>& syndrome);
    void          _init_each_cluster(int seed);
    void          _add_fault(ClusterState* cl, int j, double vw);
    ClusterState* _merge(ClusterState* cl,
                         std::unordered_set<ClusterState*>& others,
                         int connecting_j, double connecting_vw);
    ClusterState* _grow_one_step(ClusterState* cl);
    void          _get_active_valid_clusters();
};
